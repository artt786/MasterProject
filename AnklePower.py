import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
import opensim as osim

def load_mot_file(file_path):
    try:
        with open(file_path, 'r') as mot_fid:
            mot_lines = mot_fid.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise

    mot_data = mot_lines[12:]
    data = np.array([list(map(float, line.split())) for line in mot_data])
    return data


def load_trc_file(file_path, return_headers=False):
    try:
        with open(file_path, 'r') as trc_fid:
            trc_lines = trc_fid.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise

    if len(trc_lines) < 6:
        print("TRC-Datei ist zu kurz oder beschädigt.")
        return np.array([]), [], []

    # Marker-Namen extrahieren
    mheader_line = trc_lines[3].strip().split('\t')
    # Marker-Namen stehen ab Spalte 2, nur nicht-leere und nicht "Frame#" oder "Time"
    marker_names = [name for name in mheader_line[2:] if name not in ("", "Frame#", "Time")]
    n_markers = len(marker_names)
    expected_cols = 2 + 3 * n_markers
    print("Marker names:", marker_names)
    print("Number of markers:", n_markers)
    print("Expected columns:", expected_cols)

    # Koordinatenüberschriften (X1, Y1, Z1, X2, Y2, ...)
    coord_headers = trc_lines[4].strip().split('\t')

    # Datenzeilen ab Zeile 6 (Index 5)
    trc_data = trc_lines[6:]
    data_list = []

    # First, let's examine the problematic lines
    for i, line in enumerate(trc_data):
        parts = line.strip().split('\t')
        if len(parts) == expected_cols:
            try:
                # Try to convert all values to float
                row = list(map(float, parts))
                data_list.append(row)
            except ValueError as e:
                # Let's print more information about the problematic line
                print(f"Zeile {i + 7} konnte nicht konvertiert werden: {e}")
                # Print the problematic values
                problem_values = []
                for j, val in enumerate(parts):
                    try:
                        float(val)
                    except ValueError:
                        problem_values.append(f"Column {j + 1}: '{val}'")

                if problem_values:
                    print(f"  Problematische Werte: {', '.join(problem_values)}")

                # Optional: Fix-up strategy - replace problematic values with interpolated values
                # For now, we'll skip these lines, but we'll add code to interpolate later if needed
                print(f"  Diese Zeile wird übersprungen.")
        else:
            print(f"Zeile {i + 7} hat {len(parts)} Spalten, erwartet: {expected_cols} – übersprungen.")

    data = np.array(data_list)

    if data.size == 0:
        print("⚠️ Keine gültigen Datenzeilen in TRC-Datei gefunden.")

    if return_headers:
        return data, marker_names, coord_headers
    return data


def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data, axis=0)
    return filtered

def filter_trc_data(trc_data, fs=100.0, cutoff=6.0):
    marker_data = trc_data[:, 2:]
    filtered_markers = butter_lowpass_filter(marker_data, cutoff, fs)
    filtered_full = np.hstack((trc_data[:, :2], filtered_markers))
    return filtered_full

def filter_mot_data(mot_data, fs=1000.0, cutoff=20.0):
    time_col = mot_data[:, [0]]
    force_data = mot_data[:, 1:]
    filtered_forces = butter_lowpass_filter(force_data, cutoff, fs)
    filtered_full = np.hstack((time_col, filtered_forces))
    return filtered_full

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

def load_sto_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header_end = next(i for i, line in enumerate(lines) if line.strip().lower() == 'endheader')
    headers = lines[header_end + 1].strip().split()
    data = np.loadtxt(lines[header_end + 2:])
    return headers, data

# === Hauptteil ===
base_path = r"C:\\Users\\annar\\OneDrive - The University of Auckland\\Master\\ENGGEN 790\\Pilot"
mot_file = os.path.join(base_path, "walking_speed_NoAFO.mot")
trc_file = os.path.join(base_path, "transformed_walking_speed_NoAFO_markername.trc")

trc_data, marker_names, coord_headers = load_trc_file(trc_file, return_headers=True)

mot_data = load_mot_file(mot_file)

filtered_trc = filter_trc_data(trc_data, fs=100, cutoff=6)
filtered_mot = filter_mot_data(mot_data, fs=1000, cutoff=20)



# GRF anzeigen
time = mot_data[:, 0]
grf_raw = mot_data[:, 2]


# GRF anzeigen (optional, zum Vergleich)
plt.plot(time[:5000], grf_raw[:5000])
plt.title("GRF Signal (erste 5000 Frames)")
plt.xlabel("Zeit [s]")
plt.ylabel("Kraft (Spalte 16?)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Threshold auf GRF anwenden
grf_cleaned = threshold_grf(grf_raw, threshold=20.0)

# Plot nach Threshold ab Frame 800 für 5000 Frames (optional)
start_frame = 800
n_frames = 5000
plt.figure()
plt.plot(time[start_frame:start_frame + n_frames], grf_cleaned[start_frame:start_frame + n_frames], label="GRF nach Threshold")
plt.title("GRF nach Threshold-Anwendung (Frames 800–5799)")
plt.xlabel("Zeit [s]")
plt.ylabel("GRF [N]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Schrittsuche NUR mit dem thresholded Signal!
grf_cleaned_cut = grf_cleaned[start_frame:]
stance_segments_cut = find_stance_phases(grf_cleaned_cut, min_duration_frames=20)
stance_segments = [(s + start_frame, e + start_frame) for s, e in stance_segments_cut]
all_stance_times = [(time[s], time[e - 1]) for s, e in stance_segments]

# If you want ALL steps, use:
valid_stances = all_stance_times[3:8]

print(f"Gefundene Schritte insgesamt: {len(stance_segments)}")
print(f"Davon als gültig markiert: {len(valid_stances)}")


model_path = r"C:\Users\annar\OneDrive - The University of Auckland\Master\ENGGEN 790\Pilot\gait2354_simbody_scaled.osim"
model = osim.Model(model_path)
model.initSystem()


# IK für alle 5 Schritte auf einmal
ik_start_time = valid_stances[0][0]
ik_end_time = valid_stances[-1][1]

ik_output_file = os.path.join(base_path, "ik_output_all5steps.mot")
ik_settings_file = os.path.join(base_path, "IK_settings.xml")
ik_tool = osim.InverseKinematicsTool(ik_settings_file)
ik_tool.setModel(model)
ik_tool.setStartTime(ik_start_time)
ik_tool.setEndTime(ik_end_time)
ik_tool.setOutputMotionFileName(ik_output_file)
ik_tool.run()
print("IK für alle 5 Schritte abgeschlossen.")

# ID für alle 5 Schritte auf einmal
id_output_file = os.path.join(base_path, "id_output_all5steps.sto")
os.chdir(base_path)

id_tool = osim.InverseDynamicsTool()
id_tool.setModel(model)
model.initSystem()
id_tool.setCoordinatesFileName(ik_output_file)
id_tool.setExternalLoadsFileName(os.path.join(base_path, "ExLoads.xml"))
id_tool.setStartTime(ik_start_time)
id_tool.setEndTime(ik_end_time)
id_tool.setOutputGenForceFileName("id_output_all5steps.sto")
id_tool.run()
print("ID für alle 5 Schritte abgeschlossen.")

# Laden der gesamten IK- und ID-Dateien
ik_headers, ik_data = load_sto_file(ik_output_file)
id_headers, id_data = load_sto_file(id_output_file)

time_ik = ik_data[:, ik_headers.index("time")]  # Use a different variable name to avoid confusion

# Vor der Schleife
print(f"IK-Daten Zeitbereich: {time_ik[0]:.3f}s bis {time_ik[-1]:.3f}s")
print(f"Valid Stances Zeitbereiche:")
for i, (start, end) in enumerate(valid_stances):
    print(f"  Schritt {i+1}: {start:.3f}s bis {end:.3f}s")
    if start < time_ik[0] or end > time_ik[-1]:
        print(f"  WARNUNG: Schritt {i+1} liegt teilweise außerhalb des IK-Datenbereichs!")

# Jetzt kannst du für jeden Schritt die passenden Zeitbereiche extrahieren und plotten:
# Jetzt kannst du für jeden Schritt die passenden Zeitbereiche extrahieren und plotten:
for i, (start, end) in enumerate(valid_stances):
    print(f"Gültiger Schritt {i + 1}: Start = {start:.3f}s, Ende = {end:.3f}s")

    angle_idx = ik_headers.index("ankle_angle_r")
    moment_idx = id_headers.index("ankle_angle_r_moment")

    angle = ik_data[:, angle_idx]
    moment = id_data[:, moment_idx]

    # Erstelle eine Maske für den aktuellen Schritt
    mask = (time_ik >= start) & (time_ik <= end)

    if np.sum(mask) == 0:
        print(f"  WARNUNG: Keine Datenpunkte im Zeitbereich {start:.3f}s - {end:.3f}s gefunden!")
        # Versuche, den nächstgelegenen Zeitbereich zu finden
        closest_idx_start = np.argmin(np.abs(time_ik - start))
        closest_start = time_ik[closest_idx_start]
        closest_idx_end = np.argmin(np.abs(time_ik - end))
        closest_end = time_ik[closest_idx_end]

        print(f"  Nächste IK-Zeiten:")
        print(f"    Stance-Start {start:.5f}s → nächster {closest_start:.5f}s (idx {closest_idx_start})")
        print(f"    Stance-Ende {end:.5f}s → nächster {closest_end:.5f}s (idx {closest_idx_end})")

        # Erstelle angepasste Maske
        mask = (time_ik >= closest_start) & (time_ik <= closest_end)
        print(f"  Angepasste Maske: {np.sum(mask)} Datenpunkte")

        if np.sum(mask) > 0:
            print("  Verwende angepassten Zeitbereich")
            start, end = closest_start, closest_end
        else:
            print("  Konnte keinen passenden Zeitbereich finden.")
            continue

    # Extrahiere die Daten für diesen spezifischen Schritt
    step_angle = angle[mask]
    step_moment = moment[mask]
    step_time = time_ik[mask]

    # Debug information für diesen spezifischen Schritt
    print(f"\nStep {i + 1} data check:")
    print(f"  Gefundene Datenpunkte: {len(step_time)}")
    print(f"  Angle data range: {np.min(step_angle):.3f} to {np.max(step_angle):.3f}")
    print(f"  Moment data range: {np.min(step_moment):.3f} to {np.max(step_moment):.3f}")

    # Correct time step calculation
    if len(step_time) > 1:
        dt = np.mean(np.diff(step_time))
        print(f"  Average time step (dt): {dt:.5f}s")

        # Correct angular velocity calculation für diesen Schritt
        angular_velocity = np.gradient(step_angle, dt)  # Result will be in deg/s if angle is in degrees
        print(f"  Angular velocity range: {np.min(angular_velocity):.3f} to {np.max(angular_velocity):.3f} deg/s")

        # Konvertiere von Grad/s zu Rad/s für die Power-Berechnung
        angular_velocity_rad = angular_velocity * np.pi / 180

        # Power berechnen
        ankle_power = step_moment * angular_velocity_rad
        print(f"  Power range: {np.min(ankle_power):.3f} to {np.max(ankle_power):.3f} W")

        # Power plot
        plt.figure(figsize=(10, 6))
        plt.plot(step_time, ankle_power, 'b-', linewidth=2)
        plt.title(f"Ankle Power – Step {i + 1} ({start:.3f}s to {end:.3f}s)")
        plt.xlabel("Zeit [s]")
        plt.ylabel("Power [W]")
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero line
        plt.tight_layout()
        plt.show()

        # Velocity plot
        plt.figure(figsize=(10, 6))
        plt.plot(step_time, angular_velocity, 'g-', linewidth=2)
        plt.title(f"Ankle Angular Velocity – Step {i + 1} ({start:.3f}s to {end:.3f}s)")
        plt.xlabel("Zeit [s]")
        plt.ylabel("Angular Velocity [deg/s]")
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero line
        plt.tight_layout()