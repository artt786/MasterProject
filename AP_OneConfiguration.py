import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
import opensim as osim
import pandas as pd

# PTB-Imports - korrigiert basierend auf verfügbaren Klassen aus dem ersten Skript
from ptb.core import Yatsdo
from ptb.util.data import MocapDO

# Versuche andere PTB Module zu importieren (optional)
try:
    from ptb.util.io.helper import StorageIO
except ImportError:
    print("StorageIO nicht verfügbar - verwende alternative Methoden")
    StorageIO = None

try:
    from ptb.util.math.filters import Butterworth
except ImportError:
    print("PTB Butterworth Filter nicht verfügbar - verwende scipy")
    Butterworth = None

try:
    from ptb.util.math.stat import Stat
except ImportError:
    print("PTB Stat nicht verfügbar")
    Stat = None


def load_grf_from_c3d(c3d_file_path):
    """
    Lädt Ground Reaction Forces direkt aus einer C3D-Datei mit der Yac3do-Klasse

    Returns:
        time: Zeitachse
        grf_data: Dictionary mit allen GRF-Komponenten
        analog_rate: Sampling-Rate der analogen Daten
    """
    # Verwende Yac3do aus dem ersten Skript (korrigiert)
    from ptb.util.data import Yac3do  # Korrekte Klasse aus dem ersten Skript

    c3d_data = Yac3do(c3d_file_path)
    analog_data = c3d_data.analog
    analog_rate = c3d_data.c3d_dict['analog_rate']

    # Suche nach Force- und Moment-Spalten
    force_columns = [col for col in analog_data.columns if 'Force' in col or 'force' in col]
    moment_columns = [col for col in analog_data.columns if 'Moment' in col or 'moment' in col]

    print(f"Gefundene Force-Spalten: {force_columns}")
    print(f"Gefundene Moment-Spalten: {moment_columns}")
    print(f"Analog Sampling Rate: {analog_rate} Hz")

    n_samples = len(analog_data)
    time = np.array([i / analog_rate for i in range(n_samples)])

    grf_data = {
        'time': time,
        'forces': analog_data[force_columns] if force_columns else None,
        'moments': analog_data[moment_columns] if moment_columns else None,
        'all_columns': force_columns + moment_columns
    }

    return time, grf_data, analog_rate


def load_grf_from_c3d_mocapdo(c3d_file_path):
    """
    Alternative Methode mit MocapDO aus dem ersten Skript
    """

    from ptb.util.data import MocapDO

    try:
        mocap_data = MocapDO.create_from_c3d(c3d_file_path)

        if mocap_data.force_plates is not None:
            force_data = mocap_data.force_plates.data
            time = force_data['time'].values

            # Extrahiere alle Force- und Moment-Spalten
            force_columns = [col for col in force_data.columns if 'Force' in col or 'force' in col]
            moment_columns = [col for col in force_data.columns if 'Moment' in col or 'moment' in col]

            print(f"MocapDO - Gefundene Force-Spalten: {force_columns}")
            print(f"MocapDO - Gefundene Moment-Spalten: {moment_columns}")

            grf_data = {
                'time': time,
                'forces': force_data[force_columns] if force_columns else None,
                'moments': force_data[moment_columns] if moment_columns else None,
                'all_columns': force_columns + moment_columns
            }

            # Berechne Sampling-Rate aus Zeit-Daten
            analog_rate = 1.0 / np.mean(np.diff(time)) if len(time) > 1 else 1000.0

            return time, grf_data, analog_rate
        else:
            print("Keine Force Plate Daten in MocapDO gefunden")
            return None, None, None

    except Exception as e:
        print(f"Fehler mit MocapDO: {e}")
        return None, None, None


def select_grf_component(grf_data, component='Fz1'):
    """Wählt eine spezifische GRF-Komponente aus"""
    all_forces = grf_data['forces']

    if all_forces is None:
        print("Keine Force-Daten verfügbar!")
        return None

    possible_names = [
        component,
        f'Force.{component}',
        f'force.{component}',
        component.upper(),
        component.lower(),
        f'Fx1', f'Fy1', f'Fz1',  # Häufige Namen
        f'Force_Plate_1_Force_Fx',
        f'Force_Plate_1_Force_Fy',
        f'Force_Plate_1_Force_Fz'
    ]

    print(f"Verfügbare Force-Spalten: {all_forces.columns.tolist()}")

    # Exakte Übereinstimmung suchen
    for name in possible_names:
        if name in all_forces.columns:
            print(f"Verwende GRF-Komponente: {name}")
            return all_forces[name].values

    # Ähnlichkeitssuche
    for col in all_forces.columns:
        if 'fz' in col.lower() or ('z' in col.lower() and 'force' in col.lower()):
            print(f"Verwende ähnliche GRF-Komponente: {col}")
            return all_forces[col].values

    print(f"GRF-Komponente '{component}' nicht gefunden!")
    print("Verwende erste verfügbare Force-Spalte als Fallback")
    return all_forces.iloc[:, 0].values


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

    mheader_line = trc_lines[3].strip().split('\t')
    marker_names = [name for name in mheader_line[2:] if name not in ("", "Frame#", "Time")]
    n_markers = len(marker_names)
    expected_cols = 2 + 3 * n_markers
    print("Marker names:", marker_names)
    print("Number of markers:", n_markers)
    print("Expected columns:", expected_cols)

    coord_headers = trc_lines[4].strip().split('\t')
    trc_data = trc_lines[6:]
    data_list = []

    for i, line in enumerate(trc_data):
        parts = line.strip().split('\t')
        if len(parts) == expected_cols:
            try:
                row = list(map(float, parts))
                data_list.append(row)
            except ValueError as e:
                print(f"Zeile {i + 7} konnte nicht konvertiert werden: {e}")
                problem_values = []
                for j, val in enumerate(parts):
                    try:
                        float(val)
                    except ValueError:
                        problem_values.append(f"Column {j + 1}: '{val}'")
                if problem_values:
                    print(f"  Problematische Werte: {', '.join(problem_values)}")
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


def filter_grf_data(grf_array, fs=1000.0, cutoff=20.0):
    """Filtert GRF-Daten mit Butterworth-Filter"""
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


def load_sto_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header_end = next(i for i, line in enumerate(lines) if line.strip().lower() == 'endheader')
    headers = lines[header_end + 1].strip().split()
    data = np.loadtxt(lines[header_end + 2:])
    return headers, data


def filter_ik_results(ik_file_path, output_file_path, fs=100.0, cutoff=6.0):
    """Filtert IK-Ergebnisse mit Butterworth-Filter"""
    headers, data = load_sto_file(ik_file_path)
    time_col = data[:, [0]]
    coord_data = data[:, 1:]
    filtered_coords = butter_lowpass_filter(coord_data, cutoff, fs)
    filtered_data = np.hstack((time_col, filtered_coords))
    save_sto_file(output_file_path, headers, filtered_data)
    print(f"Gefilterte IK-Daten gespeichert als: {output_file_path}")
    return headers, filtered_data


def save_sto_file(file_path, headers, data):
    """Speichert Daten im .sto Format"""
    with open(file_path, 'w') as f:
        f.write("OpenSim Storage File\n")
        f.write("version=1\n")
        f.write(f"nRows={len(data)}\n")
        f.write(f"nColumns={len(headers)}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")
        f.write('\t'.join(headers) + '\n')
        for row in data:
            f.write('\t'.join([f"{val:.6f}" for val in row]) + '\n')


# === HAUPTTEIL ===
# WICHTIG: Passe diese Pfade an deine tatsächlichen Dateipfade an!
base_path = r"C:\Users\annar\OneDrive - The University of Auckland\Master\ENGGEN 790\Pilot"
c3d_file = os.path.join(base_path, "walking_speed_NoAFO.c3d")
trc_file = os.path.join(base_path, "transformed_walking_speed_NoAFO_Ella.trc")

# Überprüfe ob Dateien existieren
print("Überprüfe Dateipfade...")
if not os.path.exists(c3d_file):
    print(f"FEHLER: C3D-Datei nicht gefunden: {c3d_file}")
    print("Bitte passe den base_path an oder stelle sicher, dass die Datei existiert.")
    exit()

if not os.path.exists(trc_file):
    print(f"FEHLER: TRC-Datei nicht gefunden: {trc_file}")
    print("Bitte passe den base_path an oder stelle sicher, dass die Datei existiert.")
    exit()

print("✅ Alle Dateipfade sind korrekt!")

# TRC-Daten laden
try:
    trc_data, marker_names, coord_headers = load_trc_file(trc_file, return_headers=True)
    print(f"✅ TRC-Daten erfolgreich geladen: {len(trc_data)} Frames")
except Exception as e:
    print(f"❌ Fehler beim Laden der TRC-Datei: {e}")
    exit()

print("=" * 50)
print("LOADING GRF FROM C3D FILE")
print("=" * 50)

# Versuche zuerst die Hauptmethode mit Yac3do
grf_time = None
grf_data = None
analog_rate = None
grf_raw = None

try:
    print("Versuche Yac3do-Methode...")
    grf_time, grf_data, analog_rate = load_grf_from_c3d(c3d_file)
    grf_raw = select_grf_component(grf_data, 'Fz1')

    if grf_raw is None:
        raise Exception("Keine passende GRF-Komponente gefunden")

    print(f"✅ GRF-Daten mit Yac3do erfolgreich geladen:")
    print(f"  - Anzahl Samples: {len(grf_raw)}")
    print(f"  - Sampling Rate: {analog_rate} Hz")
    print(f"  - Zeitbereich: {grf_time[0]:.3f}s bis {grf_time[-1]:.3f}s")
    print(f"  - GRF-Bereich: {np.min(grf_raw):.1f}N bis {np.max(grf_raw):.1f}N")

except Exception as e:
    print(f"Yac3do-Methode fehlgeschlagen: {e}")
    print("Versuche MocapDO-Methode...")

    try:
        grf_time, grf_data, analog_rate = load_grf_from_c3d_mocapdo(c3d_file)

        if grf_time is not None and grf_data is not None:
            grf_raw = select_grf_component(grf_data, 'Fz1')

            if grf_raw is not None:
                print(f"✅ GRF-Daten mit MocapDO erfolgreich geladen:")
                print(f"  - Anzahl Samples: {len(grf_raw)}")
                print(f"  - Sampling Rate: {analog_rate} Hz")
                print(f"  - Zeitbereich: {grf_time[0]:.3f}s bis {grf_time[-1]:.3f}s")
                print(f"  - GRF-Bereich: {np.min(grf_raw):.1f}N bis {np.max(grf_raw):.1f}N")
            else:
                raise Exception("Keine passende GRF-Komponente mit MocapDO gefunden")
        else:
            raise Exception("MocapDO konnte keine Daten laden")

    except Exception as e2:
        print(f"Auch MocapDO-Methode schlug fehl: {e2}")
        print("Konnte keine GRF-Daten aus C3D-Datei extrahieren!")

        # Debug-Information ausgeben
        print("\n=== DEBUG-INFORMATIONEN ===")
        try:
            # Versuche wenigstens die C3D-Struktur zu analysieren
            from ptb.util.io.mocap.file_formats import Yac3do

            debug_c3d = Yac3do(c3d_file)
            print("C3D-Datei erfolgreich geöffnet")
            print(f"Verfügbare Analog-Spalten: {list(debug_c3d.analog.columns)}")
            print(f"Anzahl Analog-Samples: {len(debug_c3d.analog)}")
            print(f"Analog Rate: {debug_c3d.c3d_dict.get('analog_rate', 'Unknown')}")
        except Exception as debug_e:
            print(f"Auch Debug-Analyse fehlgeschlagen: {debug_e}")

        exit()

# Ab hier kann das Skript normal fortfahren, da grf_raw jetzt verfügbar ist

# TRC-Daten filtern
print("\nFiltering TRC data...")
filtered_trc = filter_trc_data(trc_data, fs=100, cutoff=6)

# GRF-Daten filtern
print("Filtering GRF data...")
filtered_grf = filter_grf_data(grf_raw, fs=analog_rate, cutoff=20)

# GRF anzeigen (Raw vs Filtered)
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plot_samples = min(5000, len(grf_time))
plt.plot(grf_time[:plot_samples], grf_raw[:plot_samples], 'b-', alpha=0.7, label='Raw GRF')
plt.title("Raw GRF Signal (erste 5000 Samples)")
plt.xlabel("Zeit [s]")
plt.ylabel("Kraft [N]")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(grf_time[:plot_samples], filtered_grf[:plot_samples], 'r-', alpha=0.8, label='Filtered GRF')
plt.title("Filtered GRF Signal (erste 5000 Samples)")
plt.xlabel("Zeit [s]")
plt.ylabel("Kraft [N]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Threshold auf GRF anwenden
grf_cleaned = threshold_grf(filtered_grf, threshold=20.0)

# Plot nach Threshold
start_frame = int(0.8 * analog_rate)
n_frames = min(5000, len(grf_cleaned) - start_frame)

plt.figure(figsize=(12, 6))
plt.plot(grf_time[start_frame:start_frame + n_frames],
         grf_cleaned[start_frame:start_frame + n_frames],
         label="GRF nach Threshold", linewidth=1.5)
plt.title("GRF nach Threshold-Anwendung")
plt.xlabel("Zeit [s]")
plt.ylabel("GRF [N]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Schrittsuche
print("\nSuche nach Stance Phases...")
grf_cleaned_cut = grf_cleaned[start_frame:]
stance_segments_cut = find_stance_phases(grf_cleaned_cut, min_duration_frames=int(0.02 * analog_rate))
stance_segments = [(s + start_frame, e + start_frame) for s, e in stance_segments_cut]

all_stance_times = [(grf_time[s], grf_time[e - 1]) for s, e in stance_segments]
valid_stances = all_stance_times[3:8] if len(all_stance_times) >= 8 else all_stance_times[:5]

print(f"Gefundene Schritte insgesamt: {len(stance_segments)}")
print(f"Davon als gültig markiert: {len(valid_stances)}")

for i, (start, end) in enumerate(valid_stances):
    print(f"  Schritt {i + 1}: {start:.3f}s - {end:.3f}s ({end - start:.3f}s)")

# OpenSim Analyse
model_path = r"C:\Users\annar\Documents\OpenSim\4.5\Models\Gait2354_Simbody\scaled_model_Ella_modified.osim"

# Überprüfe ob OpenSim-Modell existiert
if not os.path.exists(model_path):
    print(f"WARNUNG: OpenSim-Modell nicht gefunden: {model_path}")
    print("Bitte passe den model_path an oder stelle sicher, dass das Modell existiert.")
    print("Das Skript wird trotzdem fortgesetzt, aber OpenSim-Analysen werden übersprungen.")
    model = None
else:
    try:
        model = osim.Model(model_path)
        model.initSystem()
        print("✅ OpenSim-Modell erfolgreich geladen")
    except Exception as e:
        print(f"❌ Fehler beim Laden des OpenSim-Modells: {e}")
        model = None

if model is not None:
    print("\n=== MARKER COMPARISON ===")
    print("Model Markers:")
    model_markers = []
    marker_set = model.getMarkerSet()
    for i in range(marker_set.getSize()):
        marker_name = marker_set.get(i).getName()
        model_markers.append(marker_name)
        print(f"  {i}: '{marker_name}' (len: {len(marker_name)})")

    print(f"\nTRC Markers:")
    for i, marker_name in enumerate(marker_names):
        print(f"  {i}: '{marker_name}' (len: {len(marker_name)})")

    print(f"\nMissing in TRC:")
    for marker in model_markers:
        if marker not in marker_names:
            print(f"  '{marker}'")

    print(f"\nMissing in Model:")
    for marker in marker_names:
        if marker not in model_markers:
            print(f"  '{marker}'")

    # IK für alle 5 Schritte
    ik_start_time = valid_stances[0][0]
    ik_end_time = valid_stances[-1][1]

    ik_output_file = os.path.join(base_path, "ik_output_all5steps.mot")

    try:
        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setModel(model)
        ik_tool.setMarkerDataFileName(trc_file)
        ik_tool.setStartTime(ik_start_time)
        ik_tool.setEndTime(ik_end_time)
        ik_tool.setOutputMotionFileName(ik_output_file)

        ik_tool.run()
        print("✅ IK für alle 5 Schritte abgeschlossen.")

        # IK-Ergebnisse filtern
        filtered_ik_file = os.path.join(base_path, "ik_output_all5steps_filtered.mot")
        ik_headers_filtered, ik_data_filtered = filter_ik_results(
            ik_output_file,
            filtered_ik_file,
            fs=100.0,
            cutoff=6.0
        )
        print("✅ IK-Daten mit 6Hz Butterworth-Filter gefiltert.")

        # ID für alle 5 Schritte
        exloads_file = os.path.join(base_path, "ExLoads.xml")
        if not os.path.exists(exloads_file):
            print(f"WARNUNG: ExLoads.xml nicht gefunden: {exloads_file}")
            print("ID-Analyse wird übersprungen.")
        else:
            try:
                id_output_file = os.path.join(base_path, "id_output_all5steps.sto")
                os.chdir(base_path)

                id_tool = osim.InverseDynamicsTool()
                id_tool.setModel(model)
                model.initSystem()
                id_tool.setCoordinatesFileName(filtered_ik_file)
                id_tool.setExternalLoadsFileName(exloads_file)
                id_tool.setStartTime(ik_start_time)
                id_tool.setEndTime(ik_end_time)
                id_tool.setOutputGenForceFileName("id_output_all5steps.sto")
                id_tool.run()
                print("✅ ID für alle 5 Schritte abgeschlossen.")

                # Laden der IK- und ID-Dateien
                ik_headers, ik_data = load_sto_file(filtered_ik_file)
                id_headers, id_data = load_sto_file(id_output_file)

                time_ik = ik_data[:, ik_headers.index("time")]

                print(f"\nIK-Daten Zeitbereich: {time_ik[0]:.3f}s bis {time_ik[-1]:.3f}s")
                print(f"Valid Stances Zeitbereiche:")
                for i, (start, end) in enumerate(valid_stances):
                    print(f"  Schritt {i + 1}: {start:.3f}s bis {end:.3f}s")
                    if start < time_ik[0] or end > time_ik[-1]:
                        print(f"  WARNUNG: Schritt {i + 1} liegt teilweise außerhalb des IK-Datenbereichs!")

                # Suche nach Ankle-Daten
                ankle_angle_headers = [h for h in ik_headers if 'ankle' in h.lower() and 'angle' in h.lower()]
                ankle_moment_headers = [h for h in id_headers if 'ankle' in h.lower() and 'moment' in h.lower()]

                print("\nVerfügbare IK Headers (Ankle):")
                for i, header in enumerate(ankle_angle_headers):
                    print(f"  {ik_headers.index(header)}: {header}")

                print("\nVerfügbare ID Headers (Ankle):")
                for i, header in enumerate(ankle_moment_headers):
                    print(f"  {id_headers.index(header)}: {header}")

                if ankle_angle_headers and ankle_moment_headers:
                    # Verwende den ersten gefundenen Ankle-Header
                    angle_header = ankle_angle_headers[0]
                    moment_header = ankle_moment_headers[0]

                    angle_idx = ik_headers.index(angle_header)
                    moment_idx = id_headers.index(moment_header)

                    angle = ik_data[:, angle_idx]
                    moment = id_data[:, moment_idx]

                    print(f"\nVerwende: {angle_header} (index: {angle_idx})")
                    print(f"Verwende: {moment_header} (index: {moment_idx})")

                    # Einzelne Plots für jeden Schritt
                    print("\n" + "=" * 50)
                    print("EINZELNE PLOTS FÜR JEDEN SCHRITT")
                    print("=" * 50)

                    colors = ['red', 'blue', 'green', 'orange', 'purple']

                    for i, (start, end) in enumerate(valid_stances):
                        print(f"\nGültiger Schritt {i + 1}: Start = {start:.3f}s, Ende = {end:.3f}s")

                        mask = (time_ik >= start) & (time_ik <= end)

                        if np.sum(mask) == 0:
                            print(f"  WARNUNG: Keine Datenpunkte im Zeitbereich {start:.3f}s - {end:.3f}s gefunden!")
                            closest_idx_start = np.argmin(np.abs(time_ik - start))
                            closest_start = time_ik[closest_idx_start]
                            closest_idx_end = np.argmin(np.abs(time_ik - end))
                            closest_end = time_ik[closest_idx_end]

                            print(f"  Nächste IK-Zeiten:")
                            print(
                                f"    Stance-Start {start:.5f}s → nächster {closest_start:.5f}s (idx {closest_idx_start})")
                            print(f"    Stance-Ende {end:.5f}s → nächster {closest_end:.5f}s (idx {closest_idx_end})")

                            mask = (time_ik >= closest_start) & (time_ik <= closest_end)
                            print(f"  Angepasste Maske: {np.sum(mask)} Datenpunkte")

                            if np.sum(mask) > 0:
                                print("  Verwende angepassten Zeitbereich")
                                start, end = closest_start, closest_end
                            else:
                                print("  Konnte keinen passenden Zeitbereich finden.")
                                continue

                        step_angle = angle[mask]
                        step_moment = moment[mask]
                        step_time = time_ik[mask]

                        print(f"  Gefundene Datenpunkte: {len(step_time)}")
                        print(f"  Angle data range: {np.min(step_angle):.3f}° to {np.max(step_angle):.3f}°")
                        print(f"  Moment data range: {np.min(step_moment):.3f} to {np.max(step_moment):.3f} Nm")

                        # Ankle Angle Plot
                        plt.figure(figsize=(10, 6))
                        plt.plot(step_time, step_angle, 'r-', linewidth=2, marker='o', markersize=3)
                        plt.title(f"Ankle Angle – Step {i + 1} ({start:.3f}s to {end:.3f}s)")
                        plt.xlabel("Zeit [s]")
                        plt.ylabel("Ankle Angle [°]")
                        plt.grid(True)
                        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                        plt.tight_layout()
                        plt.show()

                        # Ankle Moment Plot
                        plt.figure(figsize=(10, 6))
                        plt.plot(step_time, step_moment, 'm-', linewidth=2)
                        plt.title(f"Ankle Moment – Step {i + 1} ({start:.3f}s to {end:.3f}s)")
                        plt.xlabel("Zeit [s]")
                        plt.ylabel("Moment [Nm]")
                        plt.grid(True)
                        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                        plt.tight_layout()
                        plt.show()

                        # Angular Velocity und Power
                        if len(step_time) > 1:
                            dt = np.mean(np.diff(step_time))
                            print(f"  Average time step (dt): {dt:.5f}s")

                            angular_velocity = np.gradient(step_angle, dt)
                            print(
                                f"  Angular velocity range: {np.min(angular_velocity):.3f} to {np.max(angular_velocity):.3f} deg/s")

                            angular_velocity_rad = angular_velocity * np.pi / 180
                            ankle_power = step_moment * angular_velocity_rad
                            print(f"  Power range: {np.min(ankle_power):.3f} to {np.max(ankle_power):.3f} W")

                            # Angular Velocity Plot
                            plt.figure(figsize=(10, 6))
                            plt.plot(step_time, angular_velocity, 'g-', linewidth=2)
                            plt.title(f"Ankle Angular Velocity – Step {i + 1} ({start:.3f}s to {end:.3f}s)")
                            plt.xlabel("Zeit [s]")
                            plt.ylabel("Angular Velocity [deg/s]")
                            plt.grid(True)
                            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                            plt.tight_layout()
                            plt.show()

                            # Power Plot
                            plt.figure(figsize=(10, 6))
                            plt.plot(step_time, ankle_power, 'b-', linewidth=2)
                            plt.title(f"Ankle Power – Step {i + 1} ({start:.3f}s to {end:.3f}s)")
                            plt.xlabel("Zeit [s]")
                            plt.ylabel("Power [W]")
                            plt.grid(True)
                            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                            plt.tight_layout()
                            plt.show()

                    # Vergleichsplots für alle Schritte
                    print("\n" + "=" * 50)
                    print("VERGLEICHSPLOTS FÜR ALLE SCHRITTE")
                    print("=" * 50)

                    plt.figure(figsize=(16, 12))

                    # Subplot 1: Alle Ankle Angles
                    plt.subplot(2, 2, 1)
                    for i, (start, end) in enumerate(valid_stances):
                        mask = (time_ik >= start) & (time_ik <= end)
                        if np.sum(mask) == 0:
                            closest_idx_start = np.argmin(np.abs(time_ik - start))
                            closest_idx_end = np.argmin(np.abs(time_ik - end))
                            mask = (time_ik >= time_ik[closest_idx_start]) & (time_ik <= time_ik[closest_idx_end])

                        step_time = time_ik[mask]
                        step_angle = angle[mask]

                        plt.plot(step_time, step_angle, color=colors[i % len(colors)],
                                 linewidth=2, label=f'Schritt {i + 1}', alpha=0.8)

                    plt.title("Ankle Angles - Alle Schritte", fontsize=14)
                    plt.xlabel("Zeit [s]")
                    plt.ylabel("Angle [°]")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Subplot 2: Alle Ankle Moments
                    plt.subplot(2, 2, 2)
                    for i, (start, end) in enumerate(valid_stances):
                        mask = (time_ik >= start) & (time_ik <= end)
                        if np.sum(mask) == 0:
                            closest_idx_start = np.argmin(np.abs(time_ik - start))
                            closest_idx_end = np.argmin(np.abs(time_ik - end))
                            mask = (time_ik >= time_ik[closest_idx_start]) & (time_ik <= time_ik[closest_idx_end])

                        step_time = time_ik[mask]
                        step_moment = moment[mask]

                        plt.plot(step_time, step_moment, color=colors[i % len(colors)],
                                 linewidth=2, label=f'Schritt {i + 1}', alpha=0.8)

                    plt.title("Ankle Moments - Alle Schritte", fontsize=14)
                    plt.xlabel("Zeit [s]")
                    plt.ylabel("Moment [Nm]")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Subplot 3: Alle Angular Velocities
                    plt.subplot(2, 2, 3)
                    for i, (start, end) in enumerate(valid_stances):
                        mask = (time_ik >= start) & (time_ik <= end)
                        if np.sum(mask) == 0:
                            closest_idx_start = np.argmin(np.abs(time_ik - start))
                            closest_idx_end = np.argmin(np.abs(time_ik - end))
                            mask = (time_ik >= time_ik[closest_idx_start]) & (time_ik <= time_ik[closest_idx_end])

                        step_time = time_ik[mask]
                        step_angle = angle[mask]

                        if len(step_time) > 1:
                            dt = np.mean(np.diff(step_time))
                            angular_velocity = np.gradient(step_angle, dt)

                            plt.plot(step_time, angular_velocity, color=colors[i % len(colors)],
                                     linewidth=2, label=f'Schritt {i + 1}', alpha=0.8)

                    plt.title("Angular Velocities - Alle Schritte", fontsize=14)
                    plt.xlabel("Zeit [s]")
                    plt.ylabel("Angular Velocity [deg/s]")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Subplot 4: Alle Powers
                    plt.subplot(2, 2, 4)
                    for i, (start, end) in enumerate(valid_stances):
                        mask = (time_ik >= start) & (time_ik <= end)
                        if np.sum(mask) == 0:
                            closest_idx_start = np.argmin(np.abs(time_ik - start))
                            closest_idx_end = np.argmin(np.abs(time_ik - end))
                            mask = (time_ik >= time_ik[closest_idx_start]) & (time_ik <= time_ik[closest_idx_end])

                        step_time = time_ik[mask]
                        step_angle = angle[mask]
                        step_moment = moment[mask]

                        if len(step_time) > 1:
                            dt = np.mean(np.diff(step_time))
                            angular_velocity = np.gradient(step_angle, dt)
                            angular_velocity_rad = angular_velocity * np.pi / 180
                            ankle_power = step_moment * angular_velocity_rad

                            plt.plot(step_time, ankle_power, color=colors[i % len(colors)],
                                     linewidth=2, label=f'Schritt {i + 1}', alpha=0.8)

                    plt.title("Ankle Power - Alle Schritte", fontsize=14)
                    plt.xlabel("Zeit [s]")
                    plt.ylabel("Power [W]")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.show()

                    # Statistiken ausgeben
                    print("\n" + "=" * 50)
                    print("ANKLE ANGLE STATISTIKEN")
                    print("=" * 50)

                    for i, (start, end) in enumerate(valid_stances):
                        mask = (time_ik >= start) & (time_ik <= end)

                        if np.sum(mask) == 0:
                            closest_idx_start = np.argmin(np.abs(time_ik - start))
                            closest_idx_end = np.argmin(np.abs(time_ik - end))
                            mask = (time_ik >= time_ik[closest_idx_start]) & (time_ik <= time_ik[closest_idx_end])

                        step_angle = angle[mask]
                        step_moment = moment[mask]
                        step_time = time_ik[mask]

                        print(f"\nSchritt {i + 1} ({start:.3f}s - {end:.3f}s):")
                        print(f"  Ankle Angle:")
                        print(f"    Minimum: {np.min(step_angle):.2f}°")
                        print(f"    Maximum: {np.max(step_angle):.2f}°")
                        print(f"    Mittelwert: {np.mean(step_angle):.2f}°")
                        print(f"    Standardabweichung: {np.std(step_angle):.2f}°")
                        print(f"    Range: {np.max(step_angle) - np.min(step_angle):.2f}°")

                        print(f"  Ankle Moment:")
                        print(f"    Minimum: {np.min(step_moment):.2f} Nm")
                        print(f"    Maximum: {np.max(step_moment):.2f} Nm")
                        print(f"    Mittelwert: {np.mean(step_moment):.2f} Nm")

                        if len(step_time) > 1:
                            dt = np.mean(np.diff(step_time))
                            angular_velocity = np.gradient(step_angle, dt)
                            angular_velocity_rad = angular_velocity * np.pi / 180
                            ankle_power = step_moment * angular_velocity_rad

                            print(f"  Angular Velocity:")
                            print(f"    Minimum: {np.min(angular_velocity):.2f} deg/s")
                            print(f"    Maximum: {np.max(angular_velocity):.2f} deg/s")

                            print(f"  Ankle Power:")
                            print(f"    Minimum: {np.min(ankle_power):.2f} W")
                            print(f"    Maximum: {np.max(ankle_power):.2f} W")
                            print(f"    Mittelwert: {np.mean(ankle_power):.2f} W")

                else:
                    print(" Kein Ankle-Daten in IK/ID-Ergebnissen gefunden!")
                    print("Verfügbare IK Headers:", ik_headers[:10], "...")
                    print("Verfügbare ID Headers:", id_headers[:10], "...")

            except Exception as e:
                print(f"Fehler bei ID-Analyse: {e}")

    except Exception as e:
        print(f"Fehler bei IK-Analyse: {e}")

else:
    print("OpenSim-Analysen übersprungen - Modell nicht verfügbar")

# GRF Visualisierung mit Stance Phases
print("\n" + "=" * 30)
print("GRF STANCE PHASES VISUALISIERUNG")
print("=" * 30)

plt.figure(figsize=(15, 8))

# Subplot 1: Gesamter GRF-Verlauf
plt.subplot(2, 1, 1)
plt.plot(grf_time, grf_cleaned, 'b-', linewidth=1, alpha=0.7, label='GRF (gefiltert & thresholded)')

# Stance Phases markieren
for i, (start_idx, end_idx) in enumerate(stance_segments[:10]):
    plt.axvspan(grf_time[start_idx], grf_time[end_idx], alpha=0.3, color=f'C{i % 10}')
    plt.text(grf_time[start_idx], np.max(grf_cleaned) * 0.9, f'S{i + 1}',
             rotation=90, fontsize=8, verticalalignment='top')

# Gültige Stance Phases hervorheben
for i, (start, end) in enumerate(valid_stances):
    plt.axvspan(start, end, alpha=0.5, color='red', label=f'Gültiger Schritt {i + 1}' if i == 0 else "")

plt.title("GRF mit markierten Stance Phases")
plt.xlabel("Zeit [s]")
plt.ylabel("GRF [N]")
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Nur gültige Stance Phases im Detail
plt.subplot(2, 1, 2)
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, (start, end) in enumerate(valid_stances):
    start_idx = np.argmin(np.abs(grf_time - start))
    end_idx = np.argmin(np.abs(grf_time - end))

    step_grf_time = grf_time[start_idx:end_idx]
    step_grf = grf_cleaned[start_idx:end_idx]

    plt.plot(step_grf_time, step_grf, color=colors[i % len(colors)],
             linewidth=2, label=f'Schritt {i + 1}', alpha=0.8)

plt.title("GRF für gültige Schritte")
plt.xlabel("Zeit [s]")
plt.ylabel("GRF [N]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

