import os
import opensim as osim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ptb.util.data import Yac3do


def find_ik_mot_files(ik_results_dir):
    """Find all MOT files from IK analysis"""
    mot_files = []
    for file in os.listdir(ik_results_dir):
        if file.endswith('.mot') and 'step_' in file:
            mot_files.append(os.path.join(ik_results_dir, file))
    mot_files.sort()
    return mot_files


def get_time_range_from_mot_file(mot_file):
    """Extract time range from MOT file"""
    with open(mot_file, 'r') as f:
        lines = f.readlines()

    # Find data start
    data_start = 0
    for i, line in enumerate(lines):
        if 'endheader' in line.lower():
            data_start = i + 2
            break
        elif 'time' in line.lower() and not line.startswith(('#', 'version', 'nRows')):
            data_start = i + 1
            break

    # Get first and last data lines
    data_lines = [line.strip() for line in lines[data_start:] if line.strip() and not line.startswith('#')]

    start_time = float(data_lines[0].split()[0])
    end_time = float(data_lines[-1].split()[0])

    return start_time, end_time


def extract_force_data_for_single_ik(c3d_file_path, start_time, end_time, padding=0.3):
    """Extract Force Plate 2 data from C3D for a single IK time range"""
    # Load C3D file
    c3d_data = Yac3do(c3d_file_path)
    analog_data = c3d_data.analog
    analog_rate = c3d_data.c3d_dict['analog_rate']

    # Find Force Plate 2 columns
    fp2_force_columns = []
    fp2_moment_columns = []

    for col in analog_data.columns:
        col_lower = col.lower()
        if ('force' in col_lower and '2' in col) or col_lower.endswith(('fx2', 'fy2', 'fz2')):
            fp2_force_columns.append(col)
        elif ('moment' in col_lower and '2' in col) or col_lower.endswith(('mx2', 'my2', 'mz2')):
            fp2_moment_columns.append(col)

    # Apply Z-rotation
    modified_analog_data = analog_data.copy()
    for col in fp2_force_columns + fp2_moment_columns:
        if 'z' in col.lower():
            modified_analog_data[col] = -modified_analog_data[col]

    # Calculate extended time range with padding
    extended_start_time = max(0.0, start_time - padding)
    extended_end_time = end_time + padding

    # Find frame indices
    start_frame = int(extended_start_time * analog_rate)
    end_frame = min(int(extended_end_time * analog_rate), len(analog_data))

    # Create time array
    n_samples = end_frame - start_frame
    step_time = np.linspace(extended_start_time, extended_end_time, n_samples)

    # Extract force/moment data
    step_forces = {}
    step_moments = {}

    for col in fp2_force_columns:
        col_lower = col.lower()
        force_data_col = modified_analog_data[col].values[start_frame:end_frame]
        if 'x' in col_lower:
            step_forces['fx'] = force_data_col
        elif 'y' in col_lower:
            step_forces['fy'] = force_data_col
        elif 'z' in col_lower:
            step_forces['fz'] = force_data_col

    for col in fp2_moment_columns:
        col_lower = col.lower()
        moment_data_col = modified_analog_data[col].values[start_frame:end_frame]
        if 'x' in col_lower:
            step_moments['mx'] = moment_data_col
        elif 'y' in col_lower:
            step_moments['my'] = moment_data_col
        elif 'z' in col_lower:
            step_moments['mz'] = moment_data_col

    return {
        'time': step_time,
        'forces': step_forces,
        'moments': step_moments,
        'cops': {'px': np.zeros_like(step_time), 'py': np.zeros_like(step_time), 'pz': np.zeros_like(step_time)},
        'start_time': extended_start_time,
        'end_time': extended_end_time
    }


def create_grf_mot_file_single(force_data, output_file, ik_start_time, ik_end_time):
    """Create GRF MOT file for a single IK matching the exact time range"""
    full_time = force_data['time']
    forces = force_data['forces']
    moments = force_data['moments']
    cops = force_data['cops']

    # Find indices for the specific IK time range
    time_mask = (full_time >= ik_start_time) & (full_time <= ik_end_time)
    time_subset = full_time[time_mask]

    # Create DataFrame
    mot_data = pd.DataFrame()
    mot_data['time'] = time_subset
    mot_data['ground_force_r_vx'] = forces.get('fx', np.zeros_like(time_subset))[time_mask]
    mot_data['ground_force_r_vy'] = forces.get('fy', np.zeros_like(time_subset))[time_mask]
    mot_data['ground_force_r_vz'] = forces.get('fz', np.zeros_like(time_subset))[time_mask]
    mot_data['ground_force_r_px'] = cops.get('px', np.zeros_like(time_subset))[time_mask]
    mot_data['ground_force_r_py'] = cops.get('py', np.zeros_like(time_subset))[time_mask]
    mot_data['ground_force_r_pz'] = cops.get('pz', np.zeros_like(time_subset))[time_mask]
    mot_data['ground_force_r_mx'] = moments.get('mx', np.zeros_like(time_subset))[time_mask]
    mot_data['ground_force_r_my'] = moments.get('my', np.zeros_like(time_subset))[time_mask]
    mot_data['ground_force_r_mz'] = moments.get('mz', np.zeros_like(time_subset))[time_mask]

    # Write MOT file
    with open(output_file, 'w') as f:
        f.write("Ground Reaction Forces\nversion=1\n")
        f.write(f"nRows={len(time_subset)}\nnColumns={len(mot_data.columns)}\n")
        f.write("inDegrees=yes\nendheader\n")
        f.write('\t'.join(mot_data.columns) + '\n')
        for _, row in mot_data.iterrows():
            f.write('\t'.join([f"{val:.6f}" for val in row.values]) + '\n')


def create_external_loads_file_simple(output_file, grf_mot_file):
    """Create External Loads XML file"""
    external_loads = osim.ExternalLoads()
    external_loads.setDataFileName(grf_mot_file)

    external_force = osim.ExternalForce()
    external_force.setName("ground_force_r")
    external_force.setAppliedToBodyName("calcn_r")
    external_force.setForceIdentifier("ground_force_r_v")
    external_force.setPointIdentifier("ground_force_r_p")
    external_force.setTorqueIdentifier("ground_force_r_m")

    external_loads.cloneAndAppend(external_force)
    external_loads.printToXML(output_file)


def create_id_setup_file(model_file, motion_file, external_loads_file, output_file, id_output_file, start_time,
                         end_time):
    """Create ID Setup XML file"""
    id_tool = osim.InverseDynamicsTool()
    id_tool.setName(f"ID_{os.path.basename(motion_file).replace('.mot', '')}")
    id_tool.setModelFileName(model_file)
    id_tool.setCoordinatesFileName(motion_file)
    id_tool.setExternalLoadsFileName(external_loads_file)

    results_dir = os.path.dirname(id_output_file)
    output_filename = os.path.basename(id_output_file)

    id_tool.setResultsDir(results_dir)
    id_tool.setOutputGenForceFileName(output_filename)
    id_tool.setStartTime(start_time)
    id_tool.setEndTime(end_time)
    id_tool.setLowpassCutoffFrequency(6.0)

    id_tool.printToXML(output_file)
    return id_tool


def run_single_inverse_dynamics(c3d_file_path, model_file, mot_file, output_base_dir):
    """Run inverse dynamics for a single IK MOT file"""
    start_time, end_time = get_time_range_from_mot_file(mot_file)
    force_data = extract_force_data_for_single_ik(c3d_file_path, start_time, end_time)

    # Create output directories
    grf_setup_dir = os.path.join(output_base_dir, "GRF_Setup")
    id_setup_dir = os.path.join(output_base_dir, "ID_Setup")
    id_results_dir = os.path.join(output_base_dir, "ID_Results")

    for dir_path in [grf_setup_dir, id_setup_dir, id_results_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Define file paths
    step_basename = os.path.basename(mot_file).replace('.mot', '')
    grf_mot_file = os.path.join(grf_setup_dir, f"{step_basename}_grf.mot")
    external_loads_file = os.path.join(grf_setup_dir, f"{step_basename}_external_loads.xml")
    id_setup_file = os.path.join(id_setup_dir, f"{step_basename}_id_setup.xml")
    id_output_file = os.path.join(id_results_dir, f"{step_basename}_id_results.sto")

    # Create files and run ID
    create_grf_mot_file_single(force_data, grf_mot_file, start_time, end_time)
    create_external_loads_file_simple(external_loads_file, grf_mot_file)
    id_tool = create_id_setup_file(model_file, mot_file, external_loads_file, id_setup_file, id_output_file, start_time,
                                   end_time)
    id_tool.run()

    return {
        'step_name': step_basename,
        'id_results_file': id_output_file,
        'start_time': start_time,
        'end_time': end_time
    }


def analyze_ankle_moments(id_results):
    """Plot ankle moments normalized to stance phase %"""
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    plt.figure(figsize=(12, 8))

    for i, result in enumerate(id_results):
        # Load STO file
        try:
            data = pd.read_csv(result['id_results_file'], sep='\t', skiprows=6)
        except:
            data = pd.read_csv(result['id_results_file'], sep='\t', skiprows=7)

        # Find ankle moment column
        ankle_moment_col = None
        for col in data.columns:
            if 'ankle' in col.lower() and 'moment' in col.lower() and 'r' in col.lower():
                ankle_moment_col = col
                break

        if ankle_moment_col:
            ankle_moment = data[ankle_moment_col].values
            stance_phase_percent = np.linspace(0, 100, len(ankle_moment))
            plt.plot(stance_phase_percent, ankle_moment, color=colors[i], linewidth=3,
                     label=f"Schritt {i + 1}", alpha=0.8)

    plt.title('Ankle Moment - Alle Schritte (% Stance Phase)', fontsize=16, fontweight='bold')
    plt.xlabel('Stance Phase [%]', fontsize=14)
    plt.ylabel('Ankle Moment [Nm]', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()


def run_id_analysis(c3d_file_path, model_file, ik_results_dir):
    """Main function to run ID analysis and plot ankle moments"""
    # Find MOT files
    mot_files = find_ik_mot_files(ik_results_dir)
    output_base_dir = os.path.dirname(ik_results_dir)

    # Run ID for each MOT file
    id_results = []
    for i, mot_file in enumerate(mot_files):
        print(f"Processing step {i + 1}: {os.path.basename(mot_file)}")
        result = run_single_inverse_dynamics(c3d_file_path, model_file, mot_file, output_base_dir)
        id_results.append(result)

    # Plot results
    analyze_ankle_moments(id_results)
    print(f"✅ Completed {len(id_results)} ID analyses")


# MAIN EXECUTION
if __name__ == "__main__":
    c3d_file_path = r"C:\Users\annar\OneDrive - The University of Auckland\Master\ENGGEN 790\Pilot\walking_speed_NoAFO.c3d"
    model_file = r"C:\Users\annar\Documents\OpenSim\4.5\Models\Gait2354_Simbody\scaled_model_Ella_modified.osim"
    ik_results_dir = os.path.join(os.path.dirname(c3d_file_path), "IK_Results")

    run_id_analysis(c3d_file_path, model_file, ik_results_dir)