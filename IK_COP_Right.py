import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
import os
import pandas as pd

from ptb.util.data import MocapDO, Yac3do
import opensim as osim


def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data, axis=0)
    return filtered


def filter_grf_data(grf_array, fs=1000.0, cutoff=20.0):
    if len(grf_array.shape) == 1:
        filtered = butter_lowpass_filter(grf_array.reshape(-1, 1), cutoff, fs)
        return filtered.flatten()
    else:
        return butter_lowpass_filter(grf_array, cutoff, fs)


def threshold_grf(grf_array, threshold=20.0):
    grf_cleaned = np.copy(grf_array)
    grf_cleaned[grf_cleaned < threshold] = 0.0001
    return grf_cleaned


def find_stance_phases(grf, min_duration_frames=10):
    stance_phases = []
    in_stance = False
    start = None
    for i, val in enumerate(grf):
        if val > 0.0001 and not in_stance:
            in_stance = True
            start = i
        elif val <= 0.0001 and in_stance:
            end = i
            if (end - start) >= min_duration_frames:
                stance_phases.append((start, end))
            in_stance = False
    if in_stance and (len(grf) - start) >= min_duration_frames:
        stance_phases.append((start, len(grf)))
    return stance_phases


def rotate_force_data_z_to_y_up(force_data, force_columns, moment_columns):
    """
    Rotate force and moment data from Z-up to Y-up coordinate system
    Step 1: Flip Z by 180 degrees (around Z-axis)
    Step 2: Z-up to Y-up rotation [-90, -90, 0] degrees
    """
    # Step 1: 180° rotation around Z-axis (flip Z)
    r1 = Rotation.from_euler('z', 180, degrees=True)
    rotation_matrix_1 = r1.as_matrix()

    # Step 2: Z-up to Y-up rotation (same as TRC)
    r2 = Rotation.from_euler('xyz', [-90, -90, 0], degrees=True)
    rotation_matrix_2 = r2.as_matrix()

    # Combined rotation: first Z flip, then Z-to-Y
    combined_rotation_matrix = np.matmul(rotation_matrix_2, rotation_matrix_1)

    rotated_data = force_data.copy()

    # Rotate force data (Fx, Fy, Fz)
    if len(force_columns) >= 3:
        # Extract force components
        fx = force_data[force_columns[0]].values
        fy = force_data[force_columns[1]].values
        fz = force_data[force_columns[2]].values

        # Stack into matrix for rotation (3 x n_samples)
        force_matrix = np.vstack([fx, fy, fz])

        # Apply combined rotation
        rotated_forces = np.matmul(combined_rotation_matrix, force_matrix)

        # Put back into dataframe
        rotated_data[force_columns[0]] = rotated_forces[0, :]
        rotated_data[force_columns[1]] = rotated_forces[1, :]
        rotated_data[force_columns[2]] = rotated_forces[2, :]

    # Rotate moment data (Mx, My, Mz)
    if len(moment_columns) >= 3:
        # Extract moment components
        mx = force_data[moment_columns[0]].values
        my = force_data[moment_columns[1]].values
        mz = force_data[moment_columns[2]].values

        # Stack into matrix for rotation (3 x n_samples)
        moment_matrix = np.vstack([mx, my, mz])

        # Apply combined rotation
        rotated_moments = np.matmul(combined_rotation_matrix, moment_matrix)

        # Put back into dataframe
        rotated_data[moment_columns[0]] = rotated_moments[0, :]
        rotated_data[moment_columns[1]] = rotated_moments[1, :]
        rotated_data[moment_columns[2]] = rotated_moments[2, :]

    return rotated_data


def load_and_process_c3d_fp2(c3d_file_path, apply_coordinate_rotation=True):
    c3d_data = Yac3do(c3d_file_path)
    analog_data = c3d_data.analog
    analog_rate = c3d_data.c3d_dict['analog_rate']

    n_samples = len(analog_data)
    time = np.array([i / analog_rate for i in range(n_samples)])

    all_columns = analog_data.columns.tolist()
    fp2_force_columns = []
    fp2_moment_columns = []

    for col in all_columns:
        col_lower = col.lower()
        if ('force' in col_lower and '2' in col) or col_lower.endswith(('fx2', 'fy2', 'fz2')):
            fp2_force_columns.append(col)
        elif ('moment' in col_lower and '2' in col) or col_lower.endswith(('mx2', 'my2', 'mz2')):
            fp2_moment_columns.append(col)

    modified_analog_data = analog_data.copy()

    # Apply coordinate system rotation if requested
    if apply_coordinate_rotation:
        modified_analog_data = rotate_force_data_z_to_y_up(
            modified_analog_data, fp2_force_columns, fp2_moment_columns
        )
        # After rotation: Z-up becomes Y-up, so vertical force is now Fy
        vertical_force_column = None
        for col in fp2_force_columns:
            if 'y' in col.lower():
                vertical_force_column = col
                break

        # Flip GRF if it's negative (multiply by -1)
        if vertical_force_column:
            modified_analog_data[vertical_force_column] = -modified_analog_data[vertical_force_column]

    else:
        # Original Z-rotation only (multiply Z components by -1)
        for col in fp2_force_columns + fp2_moment_columns:
            if 'z' in col.lower():
                modified_analog_data[col] = -modified_analog_data[col]
        # Without rotation: vertical force remains Fz
        vertical_force_column = None
        for col in fp2_force_columns:
            if 'z' in col.lower():
                vertical_force_column = col
                break

    fz2_data = modified_analog_data[vertical_force_column].values

    return c3d_data, modified_analog_data, time, fz2_data, analog_rate, fp2_force_columns, fp2_moment_columns, vertical_force_column


def create_trc_from_c3d_ptb(c3d_file_path, trc_file_path, start_time, end_time):
    try:
        mocap_data = MocapDO()
        mocap_data.load_file(c3d_file_path)

        marker_data = mocap_data.marker_data
        marker_names = mocap_data.marker_names
        point_rate = mocap_data.point_rate if hasattr(mocap_data, 'point_rate') else 100.0

        start_frame = int(start_time * point_rate)
        end_frame = int(end_time * point_rate)
        start_frame = max(0, start_frame)
        end_frame = min(len(marker_data), end_frame)

        selected_data = marker_data[start_frame:end_frame]
        n_frames = len(selected_data)

        time_column = np.array([start_time + i / point_rate for i in range(n_frames)])

        with open(trc_file_path, 'w') as f:
            f.write("PathFileType\t4\t(X/Y/Z)\t\n")
            f.write("DataType\t6\t6\t(X/Y/Z)\t\n")
            f.write("FileName\t\n")
            f.write(f"FREQUENCY\t{point_rate}\n")
            f.write(f"NO_OF_FRAMES\t{n_frames}\n")
            f.write("NO_OF_CAMERAS\t8\n")
            f.write(f"NO_OF_MARKERS\t{len(marker_names)}\n")
            f.write("UNITS\tm\n")
            f.write(f"ORIG_DATA_RATE\t{point_rate}\n")
            f.write(f"ORIG_DATA_START_FRAME\t{start_frame + 1}\n")
            f.write(f"ORIG_NO_OF_FRAMES\t{n_frames}\n")

            header1 = "Frame#\tTime\t"
            header2 = "\t\t"

            for marker in marker_names:
                header1 += f"{marker}\t\t\t"
                header2 += "X1\tY2\tZ3\t"

            f.write(header1.rstrip() + "\n")
            f.write(header2.rstrip() + "\n")

            for i in range(n_frames):
                line = f"{i + 1}\t{time_column[i]:.6f}\t"

                for j, marker in enumerate(marker_names):
                    x_val = selected_data[i, j * 3] / 1000.0 if abs(selected_data[i, j * 3]) > 10 else selected_data[
                        i, j * 3]
                    y_val = selected_data[i, j * 3 + 1] / 1000.0 if abs(selected_data[i, j * 3 + 1]) > 10 else \
                    selected_data[i, j * 3 + 1]
                    z_val = selected_data[i, j * 3 + 2] / 1000.0 if abs(selected_data[i, j * 3 + 2]) > 10 else \
                    selected_data[i, j * 3 + 2]

                    line += f"{x_val:.6f}\t{y_val:.6f}\t{z_val:.6f}\t"

                f.write(line.rstrip() + "\n")

        return True

    except Exception:
        try:
            adapter = osim.C3DFileAdapter()
            tables = adapter.read(c3d_file_path)
            markers_table = adapter.getMarkersTable(tables)
            markers_table.trim(start_time, end_time)
            osim.TRCFileAdapter.write(markers_table, trc_file_path)
            return True
        except Exception:
            return False


def create_ik_setup_file(model_file, trc_file, output_motion_file, start_time, end_time, setup_file_path):
    try:
        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setName(f"IK_Step_{start_time:.3f}s")

        try:
            ik_tool.set_model_file(model_file)
        except AttributeError:
            ik_tool.setModelFileName(model_file)

        try:
            ik_tool.set_marker_file(trc_file)
        except AttributeError:
            ik_tool.setMarkerDataFileName(trc_file)

        try:
            ik_tool.set_output_motion_file(output_motion_file)
        except AttributeError:
            ik_tool.setOutputMotionFileName(output_motion_file)

        try:
            ik_tool.set_time_range(0, start_time)
            ik_tool.set_time_range(1, end_time)
        except (AttributeError, TypeError):
            ik_tool.setStartTime(start_time)
            ik_tool.setEndTime(end_time)

        try:
            ik_tool.set_report_errors(True)
            ik_tool.set_report_marker_locations(False)
        except AttributeError:
            ik_tool.setReportErrors(True)
            ik_tool.setReportMarkerLocations(False)

        ik_tool.printToXML(setup_file_path)
        return ik_tool

    except Exception:
        xml_content = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <InverseKinematicsTool name="IK_Step_{start_time:.3f}s">
        <model_file>{model_file}</model_file>
        <marker_file>{trc_file}</marker_file>
        <coordinate_file>Unassigned</coordinate_file>
        <time_range>{start_time} {end_time}</time_range>
        <output_motion_file>{output_motion_file}</output_motion_file>
        <report_errors>true</report_errors>
        <report_marker_locations>false</report_marker_locations>
        <IKTaskSet>
            <objects />
            <groups />
        </IKTaskSet>
        <accuracy>1e-05</accuracy>
    </InverseKinematicsTool>
</OpenSimDocument>"""

        with open(setup_file_path, 'w') as f:
            f.write(xml_content)
        return True


def analyze_ik_results(mot_file_path, stance_start, stance_end):
    try:
        with open(mot_file_path, 'r') as f:
            lines = f.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            if 'endheader' in line.lower() or 'time' in line.lower():
                data_start = i + 1
                break

        data = []
        for line in lines[data_start:]:
            if line.strip():
                try:
                    values = [float(x) for x in line.strip().split()]
                    data.append(values)
                except ValueError:
                    continue

        data_array = np.array(data)
        time_col = data_array[:, 0]

        stance_mask = (time_col >= stance_start) & (time_col <= stance_end)
        stance_data = data_array[stance_mask]

        results = {
            'total_frames': len(data_array),
            'stance_frames': len(stance_data),
            'time_range': (time_col[0], time_col[-1]),
            'stance_range': (stance_start, stance_end),
            'joint_angles': stance_data[:, 1:] if stance_data.shape[1] > 1 else None
        }

        return results

    except Exception:
        return None


def run_inverse_kinematics_analysis(c3d_file_path, model_file, stance_phases, time_array, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ik_results = []
    time_extension = 0.3

    for i, (start_idx, end_idx) in enumerate(stance_phases[:5]):
        step_num = i + 1

        stance_start_time = time_array[start_idx]
        stance_end_time = time_array[end_idx - 1]

        #extended_start_time = max(0.0, stance_start_time - time_extension)
        #extended_end_time = min(time_array[-1], stance_end_time + time_extension)

        trc_file = os.path.join(output_dir, f"step_{step_num}_markers.trc")
        mot_file = os.path.join(output_dir, f"step_{step_num}_ik_results.mot")
        setup_file = os.path.join(output_dir, f"step_{step_num}_ik_setup.xml")

        try:
            success = create_trc_from_c3d_ptb(c3d_file_path, trc_file, stance_start_time, stance_end_time)

            if not success:
                continue

            ik_tool = create_ik_setup_file(model_file, trc_file, mot_file, stance_start_time, stance_end_time,
                                           setup_file)

            if ik_tool is None:
                continue

            if hasattr(ik_tool, 'run') and callable(getattr(ik_tool, 'run')):
                ik_tool.run()
            elif ik_tool == True:
                import subprocess
                cmd = f"opensim-cmd run-tool {setup_file}"
                subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if os.path.exists(mot_file):
                ik_data = analyze_ik_results(mot_file, stance_start_time, stance_end_time)
                ik_results.append({
                    'step': step_num,
                    'stance_start': stance_start_time,
                    'stance_end': stance_end_time,
                    'duration': stance_end_time - stance_start_time,
                    'start_frame': start_idx,
                    'end_frame': end_idx - 1,
                    'mot_file': mot_file,
                    'trc_file': trc_file,
                    'setup_file': setup_file,
                    'ik_data': ik_data
                })

        except Exception:
            continue

    return ik_results


def plot_grf_analysis(time, fz2_raw, fz2_filtered, fz2_thresholded, stance_phases, search_start_frame, analog_rate,
                      force_component="Fz2"):
    plot_samples = min(5000, len(time))

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time[:plot_samples], fz2_raw[:plot_samples], 'b-', alpha=0.7, label=f'Raw {force_component} (rotated)',
             linewidth=1)
    plt.plot(time[:plot_samples], fz2_filtered[:plot_samples], 'r-', alpha=0.8,
             label=f'Filtered {force_component} (20Hz)', linewidth=1.5)
    plt.title(f"Force Plate 2: Raw vs Filtered GRF ({force_component}, first 5000 frames)")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(time[:plot_samples], fz2_filtered[:plot_samples], 'r-', alpha=0.8, label=f'Filtered {force_component}',
             linewidth=1)
    plt.plot(time[:plot_samples], fz2_thresholded[:plot_samples], 'g-', alpha=0.8,
             label=f'Thresholded {force_component} (20N)', linewidth=1.5)
    plt.title(f"Filtered vs Thresholded GRF ({force_component})")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(time, fz2_thresholded, 'g-', alpha=0.7, label=f'Thresholded {force_component}', linewidth=1)

    search_start_time = search_start_frame / analog_rate
    plt.axvline(x=search_start_time, color='black', linestyle='--', linewidth=2,
                label=f'Search start (Frame {search_start_frame})')

    for i, (start_idx, end_idx) in enumerate(stance_phases[:5]):
        plt.axvspan(time[start_idx], time[end_idx - 1], alpha=0.6, facecolor='red',
                    edgecolor='black', linewidth=2, label=f'Step {i + 1}' if i == 0 else '')

    plt.title(f"Identified Stance Phases (search from frame {search_start_frame}, first 5 highlighted)")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_individual_steps(valid_stances, stance_phases, time, fz2_thresholded, force_component="Fz2"):
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    for i, (start_time, end_time) in enumerate(valid_stances):
        start_idx, end_idx = stance_phases[i]
        step_time = time[start_idx:end_idx]
        step_grf = fz2_thresholded[start_idx:end_idx]
        step_time_normalized = np.linspace(0, 100, len(step_time))

        plt.plot(step_time_normalized, step_grf, color=colors[i], linewidth=2,
                 label=f'Step {i + 1}', alpha=0.8)

    plt.title(f"Ground Reaction Force - All 5 Steps (normalized to % stance phase) - {force_component}")
    plt.xlabel("Stance Phase [%]")
    plt.ylabel("Vertical GRF [N]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)

    plt.subplot(2, 1, 2)
    for i, (start_time, end_time) in enumerate(valid_stances):
        start_idx, end_idx = stance_phases[i]
        step_time = time[start_idx:end_idx]
        step_grf = fz2_thresholded[start_idx:end_idx]

        plt.plot(step_time, step_grf, color=colors[i], linewidth=2,
                 label=f'Step {i + 1}', alpha=0.8)

    plt.title(f"Ground Reaction Force - All 5 Steps (absolute time) - {force_component}")
    plt.xlabel("Time [s]")
    plt.ylabel("Vertical GRF [N]")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (start_time, end_time) in enumerate(valid_stances):
        if i < 5:
            start_idx, end_idx = stance_phases[i]
            step_time = time[start_idx:end_idx]
            step_grf = fz2_thresholded[start_idx:end_idx]
            step_time_normalized = np.linspace(0, 100, len(step_time))

            axes[i].plot(step_time_normalized, step_grf, color=colors[i], linewidth=2)
            axes[i].set_title(
                f'Step {i + 1}\n{start_time:.3f}s - {end_time:.3f}s (Duration: {end_time - start_time:.3f}s)')
            axes[i].set_xlabel('Stance Phase [%]')
            axes[i].set_ylabel('Vertical GRF [N]')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, 100)

            max_grf = np.max(step_grf)
            mean_grf = np.mean(step_grf)
            axes[i].text(0.05, 0.95, f'Max: {max_grf:.0f}N\nMean: {mean_grf:.0f}N',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    axes[5].axis('off')
    plt.suptitle(f'Ground Reaction Force - Individual Steps (Force Plate 2) - {force_component}', fontsize=16)
    plt.tight_layout()
    plt.show()


def run_gait_analysis(c3d_file_path, model_file, search_start_frame=7500, apply_coordinate_rotation=True):
    c3d_data, modified_analog_data, time, fz2_data, analog_rate, fp2_force_cols, fp2_moment_cols, vertical_force_column = load_and_process_c3d_fp2(
        c3d_file_path, apply_coordinate_rotation=apply_coordinate_rotation
    )

    fz2_raw = fz2_data.copy()
    fz2_filtered = filter_grf_data(fz2_raw, fs=analog_rate, cutoff=20.0)
    fz2_thresholded = threshold_grf(fz2_filtered, threshold=20.0)

    min_duration_frames = int(0.02 * analog_rate)
    fz2_search = fz2_thresholded[search_start_frame:]
    stance_phases_relative = find_stance_phases(fz2_search, min_duration_frames=min_duration_frames)
    stance_phases = [(start + search_start_frame, end + search_start_frame) for start, end in stance_phases_relative]

    stance_times = []
    for start_idx, end_idx in stance_phases:
        start_time = time[start_idx]
        end_time = time[end_idx - 1]
        stance_times.append((start_time, end_time))

    valid_stances = stance_times[:5]

    # Update plot titles to reflect the correct force component
    force_component_name = "Fy2" if apply_coordinate_rotation else "Fz2"

    plot_grf_analysis(time, fz2_raw, fz2_filtered, fz2_thresholded, stance_phases, search_start_frame, analog_rate,
                      force_component_name)
    plot_individual_steps(valid_stances, stance_phases, time, fz2_thresholded, force_component_name)

    output_dir = os.path.join(os.path.dirname(c3d_file_path), "IK_Results")
    ik_results = run_inverse_kinematics_analysis(c3d_file_path, model_file, stance_phases, time, output_dir)

    return {
        'stance_phases': stance_phases,
        'valid_stances': valid_stances,
        'ik_results': ik_results,
        'grf_data': {
            'time': time,
            'raw': fz2_raw,
            'filtered': fz2_filtered,
            'thresholded': fz2_thresholded
        }
    }


if __name__ == "__main__":
    c3d_file_path = r"C:\Users\annar\OneDrive - The University of Auckland\Master\ENGGEN 790\Pilot\walking_speed_NoAFO.c3d"
    model_file = r"C:\Users\annar\Documents\OpenSim\4.5\Models\Gait2354_Simbody\scaled_model_Ella_modified.osim"

    # Run with coordinate rotation (default)
    results = run_gait_analysis(c3d_file_path, model_file, apply_coordinate_rotation=True)

    # Or run without coordinate rotation (original method)
    # results = run_gait_analysis(c3d_file_path, model_file, apply_coordinate_rotation=False)