import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yatpkg.util.data import TRC, MocapDO
from yatpkg.math.filters import Butterworth
from yatpkg.math.transformation import Cloud
import copy
import shutil
import ezc3d


def main():
    # Pfade für die Dateien
    c3d_path = r'C:\Users\annar\OneDrive - The University of Auckland\Master\ENGGEN 790\Treadmill\New Session\10C3D_Left'
    output_path = r'C:\Users\annar\OneDrive - The University of Auckland\Master\ENGGEN 790\Treadmill\New Session\CalibratedForcePlates_Left10'

    # Erstellen des Ausgabeordners, falls er nicht existiert
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Einlesen aller C3D-Dateien
    c3d_files = [f for f in os.listdir(c3d_path) if f.startswith('LeftFP_') and f.endswith('.c3d')]
    c3d_files.sort()

    print(f"Gefundene C3D-Dateien: {len(c3d_files)}")

    # Sammeln der Markerpositionen aus allen Dateien
    marker_positions = {}

    for c3d_file in c3d_files:
        file_base = c3d_file.replace('.c3d', '')
        print(f"Verarbeite Datei: {c3d_file}")

        try:
            # Laden der C3D-Datei
            mocap = MocapDO.create_from_c3d(os.path.join(c3d_path, c3d_file))

            # WICHTIG: Nicht zu Y-Up konvertieren für Kraftplatten-Kalibrierung
            # mocap.markers.z_up_to_y_up()  # Kommentiert aus

            # Speichern der Markerpositionen (im ursprünglichen Koordinatensystem)
            marker_positions[file_base] = {}
            for marker_name in ['RightF', 'RightB', 'LeftF', 'LeftB']:
                try:
                    marker_positions[file_base][marker_name] = np.nanmean(mocap.markers.marker_set[marker_name], axis=0)
                except KeyError:
                    print(f"Warnung: Marker {marker_name} in {file_base} nicht gefunden!")
                    marker_positions[file_base][marker_name] = np.array([np.nan, np.nan, np.nan])

        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {c3d_file}: {e}")

    # Berechnen der mittleren Markerpositionen
    mean_markers = calculate_mean_markers(marker_positions)
    print("Mittlere Markerpositionen berechnet.")

    # Ausgabe der mittleren Markerpositionen zur Überprüfung
    for marker, pos in mean_markers.items():
        print(f"{marker}: {pos}")

    # Lese die aktuellen Kraftplatten-Parameter aus einer Beispieldatei
    example_c3d = os.path.join(c3d_path, c3d_files[0])
    current_corners, current_origins = read_current_forceplate_params(example_c3d)

    print("Aktuelle Kraftplatten-Parameter:")
    if current_corners is not None:
        print(f"Corners shape: {current_corners.shape}")
        print(f"Origins shape: {current_origins.shape}")

    # Berechne neue Kraftplatten-Eckpunkte (mit ursprünglichen Koordinaten)
    new_fp4_corners, new_fp5_corners = calculate_forceplate_corners_fixed(mean_markers)

    # Optional: Z-Koordinaten-Korrektur für COP-Problem
    print("\n=== Z-KOORDINATEN KORREKTUR ===")
    print("Falls COP nach unten zeigt, können Z-Koordinaten angepasst werden:")

    # Alle Eckpunkte auf dieselbe Z-Höhe setzen (Durchschnitt)
    all_z_coords = np.concatenate([new_fp4_corners[:, 2], new_fp5_corners[:, 2]])
    average_z = np.mean(all_z_coords)

    print(f"Durchschnittliche Z-Höhe: {average_z:.2f}")
    print("Setze alle Kraftplatten-Eckpunkte auf diese Höhe...")

    # Optional: Uncomment die nächsten Zeilen, falls COP-Problem weiterhin besteht
    # new_fp4_corners[:, 2] = average_z
    # new_fp5_corners[:, 2] = average_z

    print("\n=== VERWENDETE KOORDINATEN ===")
    print("Verwende die ursprünglichen gespeicherten Marker-Koordinaten direkt für die Kraftplatten:")
    for marker, pos in mean_markers.items():
        print(f"{marker}: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")

    print("Neue Kraftplatteneckpunkte berechnet:")
    print("FP4 Eckpunkte (Kraftplatte 4, Index 3):", new_fp4_corners)
    print("FP5 Eckpunkte (Kraftplatte 5, Index 4):", new_fp5_corners)

    print(f"\nVerwende KORRIGIERTE umfassende Kraftplatten-Korrektur für COP-Problem")

    # Aktualisiere alle C3D-Dateien mit der KORRIGIERTEN umfassenden Korrektur
    for c3d_file in c3d_files:
        input_c3d = os.path.join(c3d_path, c3d_file)
        output_c3d = os.path.join(output_path, f'Calibrated_{c3d_file}')

        print(f"\nVerarbeite: {c3d_file}")
        success = comprehensive_forceplate_correction_fixed(input_c3d, output_c3d, [new_fp4_corners, new_fp5_corners])

        if not success:
            print(f"Verwende Fallback-Methode für {c3d_file}")
            update_forceplate_corners_in_c3d_fixed(input_c3d, output_c3d, [new_fp4_corners, new_fp5_corners])
        else:
            # Validiere die COP-Transformation
            validate_cop_transformation(output_c3d, [3, 4])

    # Visualisieren der Ergebnisse
    visualize_results_fixed(mean_markers, new_fp4_corners, new_fp5_corners)

    print("Kalibrierung abgeschlossen!")


def read_current_forceplate_params(c3d_path):
    """Liest die aktuellen Kraftplatten-Parameter aus einer C3D-Datei."""
    try:
        c3d = ezc3d.c3d(c3d_path)
        corners = c3d['parameters']['FORCE_PLATFORM']['CORNERS']['value']
        origins = c3d['parameters']['FORCE_PLATFORM']['ORIGIN']['value']
        return corners, origins
    except Exception as e:
        print(f"Fehler beim Lesen der Kraftplatten-Parameter: {e}")
        return None, None


def comprehensive_forceplate_correction_fixed(input_c3d_path, output_c3d_path, new_corners_list):
    """
    Umfassende Korrektur der Kraftplatten-Daten inkl. korrekter COP-Transformation.
    """
    try:
        c3d = ezc3d.c3d(input_c3d_path)

        # 1. Kraftplatten-Indizes
        fp4_index = 3  # Kraftplatte 4
        fp5_index = 4  # Kraftplatte 5

        print("=== KRAFTPLATTEN-KORREKTUR ===")

        # 2. Alte Geometrie speichern für Transformation
        old_corners = c3d['parameters']['FORCE_PLATFORM']['CORNERS']['value'].copy()
        old_origins = c3d['parameters']['FORCE_PLATFORM']['ORIGIN']['value'].copy()

        # Alte Eckpunkte und Ursprünge für FP4 und FP5
        old_fp4_corners = old_corners[:, :, fp4_index].T
        old_fp5_corners = old_corners[:, :, fp5_index].T
        old_fp4_origin = old_origins[:, fp4_index]
        old_fp5_origin = old_origins[:, fp5_index]

        # 3. Neue Geometrie setzen
        current_corners = old_corners.copy()
        current_origins = old_origins.copy()

        # Neue Eckpunkte setzen
        current_corners[:, :, fp4_index] = new_corners_list[0].T
        current_corners[:, :, fp5_index] = new_corners_list[1].T
        c3d['parameters']['FORCE_PLATFORM']['CORNERS']['value'] = current_corners

        # Neue Ursprünge berechnen
        fp4_origin = np.mean(new_corners_list[0], axis=0)
        fp5_origin = np.mean(new_corners_list[1], axis=0)
        current_origins[:, fp4_index] = fp4_origin
        current_origins[:, fp5_index] = fp5_origin
        c3d['parameters']['FORCE_PLATFORM']['ORIGIN']['value'] = current_origins

        print(f"Kraftplatte 4:")
        print(f"  Alter Origin: {old_fp4_origin}")
        print(f"  Neuer Origin: {fp4_origin}")
        print(f"Kraftplatte 5:")
        print(f"  Alter Origin: {old_fp5_origin}")
        print(f"  Neuer Origin: {fp5_origin}")

        # 4. Berechne Transformationsmatrizen für jede Kraftplatte
        def calculate_transformation_matrix(old_corners, new_corners, old_origin, new_origin):
            """Berechnet die Transformationsmatrix von alter zu neuer Kraftplatte."""
            # Alte lokale Koordinatensystem
            old_x = old_corners[1] - old_corners[0]
            old_x = old_x / np.linalg.norm(old_x)
            old_y = old_corners[3] - old_corners[0]
            old_y = old_y / np.linalg.norm(old_y)
            old_z = np.cross(old_x, old_y)
            old_z = old_z / np.linalg.norm(old_z)

            # Neues lokales Koordinatensystem
            new_x = new_corners[1] - new_corners[0]
            new_x = new_x / np.linalg.norm(new_x)
            new_y = new_corners[3] - new_corners[0]
            new_y = new_y / np.linalg.norm(new_y)
            new_z = np.cross(new_x, new_y)
            new_z = new_z / np.linalg.norm(new_z)

            # Rotationsmatrix von alt zu neu
            old_rotation = np.column_stack([old_x, old_y, old_z])
            new_rotation = np.column_stack([new_x, new_y, new_z])
            rotation_matrix = new_rotation @ old_rotation.T

            # Translation
            translation = new_origin - old_origin

            return rotation_matrix, translation

        # Transformationsmatrizen berechnen
        fp4_rotation, fp4_translation = calculate_transformation_matrix(
            old_fp4_corners, new_corners_list[0], old_fp4_origin, fp4_origin)
        fp5_rotation, fp5_translation = calculate_transformation_matrix(
            old_fp5_corners, new_corners_list[1], old_fp5_origin, fp5_origin)

        # 5. Analoge Daten korrigieren
        if 'data' in c3d and 'analogs' in c3d['data']:
            analog_data = c3d['data']['analogs'].copy()
            analog_labels = c3d['parameters']['ANALOG']['LABELS']['value']

            print("\n=== ANALOGE DATEN KORREKTUR ===")

            # Finde alle relevanten Kanäle für Kraftplatten 4 und 5
            fp4_channels = {}
            fp5_channels = {}

            for i, label in enumerate(analog_labels):
                label_upper = label.upper().strip()

                # Kraftplatte 4 Kanäle
                if any(identifier in label_upper for identifier in
                       ['FP4', 'PLATE4', '4FX', '4FY', '4FZ', '4MX', '4MY', '4MZ', '4COPX', '4COPY', '4COPZ']):
                    if 'FX' in label_upper or ('FORCE' in label_upper and 'X' in label_upper):
                        fp4_channels['fx'] = i
                    elif 'FY' in label_upper or ('FORCE' in label_upper and 'Y' in label_upper):
                        fp4_channels['fy'] = i
                    elif 'FZ' in label_upper or ('FORCE' in label_upper and 'Z' in label_upper):
                        fp4_channels['fz'] = i
                    elif 'MX' in label_upper or ('MOMENT' in label_upper and 'X' in label_upper):
                        fp4_channels['mx'] = i
                    elif 'MY' in label_upper or ('MOMENT' in label_upper and 'Y' in label_upper):
                        fp4_channels['my'] = i
                    elif 'MZ' in label_upper or ('MOMENT' in label_upper and 'Z' in label_upper):
                        fp4_channels['mz'] = i
                    elif 'COPX' in label_upper or ('COP' in label_upper and 'X' in label_upper):
                        fp4_channels['copx'] = i
                    elif 'COPY' in label_upper or ('COP' in label_upper and 'Y' in label_upper):
                        fp4_channels['copy'] = i
                    elif 'COPZ' in label_upper or ('COP' in label_upper and 'Z' in label_upper):
                        fp4_channels['copz'] = i

                # Kraftplatte 5 Kanäle
                elif any(identifier in label_upper for identifier in
                         ['FP5', 'PLATE5', '5FX', '5FY', '5FZ', '5MX', '5MY', '5MZ', '5COPX', '5COPY', '5COPZ']):
                    if 'FX' in label_upper or ('FORCE' in label_upper and 'X' in label_upper):
                        fp5_channels['fx'] = i
                    elif 'FY' in label_upper or ('FORCE' in label_upper and 'Y' in label_upper):
                        fp5_channels['fy'] = i
                    elif 'FZ' in label_upper or ('FORCE' in label_upper and 'Z' in label_upper):
                        fp5_channels['fz'] = i
                    elif 'MX' in label_upper or ('MOMENT' in label_upper and 'X' in label_upper):
                        fp5_channels['mx'] = i
                    elif 'MY' in label_upper or ('MOMENT' in label_upper and 'Y' in label_upper):
                        fp5_channels['my'] = i
                    elif 'MZ' in label_upper or ('MOMENT' in label_upper and 'Z' in label_upper):
                        fp5_channels['mz'] = i
                    elif 'COPX' in label_upper or ('COP' in label_upper and 'X' in label_upper):
                        fp5_channels['copx'] = i
                    elif 'COPY' in label_upper or ('COP' in label_upper and 'Y' in label_upper):
                        fp5_channels['copy'] = i
                    elif 'COPZ' in label_upper or ('COP' in label_upper and 'Z' in label_upper):
                        fp5_channels['copz'] = i

            print(f"FP4 Kanäle gefunden: {fp4_channels}")
            print(f"FP5 Kanäle gefunden: {fp5_channels}")

            # 6. Transformiere COP-Daten korrekt
            def transform_cop_data(channels, rotation_matrix, translation, plate_name, old_origin, new_origin):
                """Transformiert COP-Daten korrekt basierend auf der Kraftplatten-Verschiebung."""
                print(f"\n--- {plate_name} COP Transformation ---")

                if not all(key in channels for key in ['copx', 'copy', 'copz']):
                    print(f"Warnung: Nicht alle COP-Kanäle für {plate_name} gefunden!")
                    return

                copx_idx = channels['copx']
                copy_idx = channels['copy']
                copz_idx = channels['copz']

                # Originale COP-Daten
                original_copx = analog_data[copx_idx, :].copy()
                original_copy = analog_data[copy_idx, :].copy()
                original_copz = analog_data[copz_idx, :].copy()

                # Nur nicht-null Werte transformieren
                valid_mask = (original_copx != 0) | (original_copy != 0) | (original_copz != 0)

                if not np.any(valid_mask):
                    print(f"Keine gültigen COP-Daten für {plate_name} gefunden!")
                    return

                print(f"Transformiere {np.sum(valid_mask)} COP-Datenpunkte für {plate_name}")

                # COP-Punkte als Array organisieren
                n_frames = analog_data.shape[1]
                cop_points = np.zeros((n_frames, 3))
                cop_points[:, 0] = original_copx
                cop_points[:, 1] = original_copy
                cop_points[:, 2] = original_copz

                # Transformiere jeden gültigen COP-Punkt
                for i in range(n_frames):
                    if valid_mask[i]:
                        # Relative Position zur alten Kraftplatte
                        relative_pos = cop_points[i] - old_origin

                        # Transformiere die relative Position
                        transformed_relative = rotation_matrix @ relative_pos

                        # Neue absolute Position
                        cop_points[i] = new_origin + transformed_relative

                # Aktualisierte Daten zurückschreiben
                analog_data[copx_idx, :] = cop_points[:, 0]
                analog_data[copy_idx, :] = cop_points[:, 1]
                analog_data[copz_idx, :] = cop_points[:, 2]

                print(f"{plate_name} COP-Transformation abgeschlossen")

                # Statistiken zur Überprüfung
                valid_new_cop = cop_points[valid_mask]
                if len(valid_new_cop) > 0:
                    print(f"  COP Bereich X: {valid_new_cop[:, 0].min():.1f} bis {valid_new_cop[:, 0].max():.1f}")
                    print(f"  COP Bereich Y: {valid_new_cop[:, 1].min():.1f} bis {valid_new_cop[:, 1].max():.1f}")
                    print(f"  COP Bereich Z: {valid_new_cop[:, 2].min():.1f} bis {valid_new_cop[:, 2].max():.1f}")

            # Transformiere COP für beide Kraftplatten
            if fp4_channels:
                transform_cop_data(fp4_channels, fp4_rotation, fp4_translation, "FP4", old_fp4_origin, fp4_origin)
            if fp5_channels:
                transform_cop_data(fp5_channels, fp5_rotation, fp5_translation, "FP5", old_fp5_origin, fp5_origin)

            # Aktualisierte analoge Daten zurückschreiben
            c3d['data']['analogs'] = analog_data

        # 7. Weitere Parameter aktualisieren
        if 'CAL_MATRIX' in c3d['parameters']['FORCE_PLATFORM']:
            cal_matrix = c3d['parameters']['FORCE_PLATFORM']['CAL_MATRIX']['value']
            print(f"CAL_MATRIX aktualisiert")

            if len(cal_matrix.shape) == 3 and cal_matrix.shape[2] > max(fp4_index, fp5_index):
                # Rotationsmatrizen aktualisieren
                if cal_matrix.shape[0] >= 3 and cal_matrix.shape[1] >= 3:
                    # FP4
                    new_x = new_corners_list[0][1] - new_corners_list[0][0]
                    new_x = new_x / np.linalg.norm(new_x)
                    new_y = new_corners_list[0][3] - new_corners_list[0][0]
                    new_y = new_y / np.linalg.norm(new_y)
                    new_z = np.cross(new_x, new_y)
                    if new_z[2] < 0:
                        new_z = -new_z
                    new_z = new_z / np.linalg.norm(new_z)

                    rotation_fp4 = np.column_stack([new_x, new_y, new_z])
                    cal_matrix[0:3, 0:3, fp4_index] = rotation_fp4

                    # FP5
                    new_x = new_corners_list[1][1] - new_corners_list[1][0]
                    new_x = new_x / np.linalg.norm(new_x)
                    new_y = new_corners_list[1][3] - new_corners_list[1][0]
                    new_y = new_y / np.linalg.norm(new_y)
                    new_z = np.cross(new_x, new_y)
                    if new_z[2] < 0:
                        new_z = -new_z
                    new_z = new_z / np.linalg.norm(new_z)

                    rotation_fp5 = np.column_stack([new_x, new_y, new_z])
                    cal_matrix[0:3, 0:3, fp5_index] = rotation_fp5

                c3d['parameters']['FORCE_PLATFORM']['CAL_MATRIX']['value'] = cal_matrix

        # 8. ZERO Parameter zurücksetzen
        if 'ZERO' in c3d['parameters']['FORCE_PLATFORM']:
            zeros = c3d['parameters']['FORCE_PLATFORM']['ZERO']['value']
            if len(zeros.shape) >= 2 and zeros.shape[1] > max(fp4_index, fp5_index):
                zeros[:, fp4_index] = 0
                zeros[:, fp5_index] = 0
                c3d['parameters']['FORCE_PLATFORM']['ZERO']['value'] = zeros

        # 9. Datei speichern
        c3d.write(output_c3d_path)
        print(f"\n✅ Korrigierte C3D-Datei gespeichert: {output_c3d_path}")
        print("COP wurde korrekt mit den Kraftplatten transformiert!")

        return True

    except Exception as e:
        print(f"❌ Fehler bei der Korrektur: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_cop_transformation(c3d_path, fp_indices=[3, 4]):
    """
    Validiert die COP-Transformation in einer korrigierten C3D-Datei.
    """
    try:
        c3d = ezc3d.c3d(c3d_path)

        # Kraftplatten-Geometrie
        corners = c3d['parameters']['FORCE_PLATFORM']['CORNERS']['value']
        origins = c3d['parameters']['FORCE_PLATFORM']['ORIGIN']['value']

        # Analoge Daten
        analog_data = c3d['data']['analogs']
        analog_labels = c3d['parameters']['ANALOG']['LABELS']['value']

        print(f"\n=== COP-VALIDIERUNG für {os.path.basename(c3d_path)} ===")

        for fp_idx in fp_indices:
            plate_name = f"FP{fp_idx + 1}"
            print(f"\n{plate_name} (Index {fp_idx}):")

            # Kraftplatten-Grenzen
            plate_corners = corners[:, :, fp_idx].T
            min_x, max_x = plate_corners[:, 0].min(), plate_corners[:, 0].max()
            min_y, max_y = plate_corners[:, 1].min(), plate_corners[:, 1].max()
            min_z, max_z = plate_corners[:, 2].min(), plate_corners[:, 2].max()

            print(f"  Kraftplatte Grenzen:")
            print(f"    X: {min_x:.1f} bis {max_x:.1f}")
            print(f"    Y: {min_y:.1f} bis {max_y:.1f}")
            print(f"    Z: {min_z:.1f} bis {max_z:.1f}")

            # Finde COP-Kanäle
            cop_channels = {}
            for i, label in enumerate(analog_labels):
                label_upper = label.upper().strip()
                if str(fp_idx + 1) in label or f"FP{fp_idx + 1}" in label_upper:
                    if 'COPX' in label_upper:
                        cop_channels['x'] = i
                    elif 'COPY' in label_upper:
                        cop_channels['y'] = i
                    elif 'COPZ' in label_upper:
                        cop_channels['z'] = i

            if len(cop_channels) == 3:
                copx_data = analog_data[cop_channels['x'], :]
                copy_data = analog_data[cop_channels['y'], :]
                copz_data = analog_data[cop_channels['z'], :]

                # Nur gültige COP-Punkte analysieren
                valid_mask = (copx_data != 0) | (copy_data != 0) | (copz_data != 0)

                if np.any(valid_mask):
                    valid_copx = copx_data[valid_mask]
                    valid_copy = copy_data[valid_mask]
                    valid_copz = copz_data[valid_mask]

                    print(f"  COP-Daten ({np.sum(valid_mask)} gültige Punkte):")
                    print(f"    X: {valid_copx.min():.1f} bis {valid_copx.max():.1f}")
                    print(f"    Y: {valid_copy.min():.1f} bis {valid_copy.max():.1f}")
                    print(f"    Z: {valid_copz.min():.1f} bis {valid_copz.max():.1f}")

                    # Prüfe, ob COP innerhalb der Kraftplatte liegt
                    x_in_bounds = (valid_copx >= min_x - 50) & (valid_copx <= max_x + 50)  # 50mm Toleranz
                    y_in_bounds = (valid_copy >= min_y - 50) & (valid_copy <= max_y + 50)
                    z_reasonable = (valid_copz >= min_z - 50) & (valid_copz <= max_z + 50)

                    print(f"  COP-Validierung:")
                    print(
                        f"    X in Grenzen: {np.sum(x_in_bounds)}/{len(valid_copx)} ({100 * np.sum(x_in_bounds) / len(valid_copx):.1f}%)")
                    print(
                        f"    Y in Grenzen: {np.sum(y_in_bounds)}/{len(valid_copy)} ({100 * np.sum(y_in_bounds) / len(valid_copy):.1f}%)")
                    print(
                        f"    Z vernünftig: {np.sum(z_reasonable)}/{len(valid_copz)} ({100 * np.sum(z_reasonable) / len(valid_copz):.1f}%)")
                else:
                    print(f"  Keine gültigen COP-Daten gefunden!")
            else:
                print(f"  COP-Kanäle nicht vollständig gefunden: {cop_channels}")

    except Exception as e:
        print(f"Fehler bei COP-Validierung: {e}")


def calculate_mean_markers(marker_positions):
    """Berechnet die mittleren Positionen aller Marker über alle gültigen Dateien."""
    mean_markers = {}
    valid_counts = {}

    for marker_name in ['RightF', 'RightB', 'LeftF', 'LeftB']:
        mean_markers[marker_name] = np.zeros(3)
        valid_counts[marker_name] = 0

    for file_base, markers in marker_positions.items():
        for marker_name, position in markers.items():
            if not np.any(np.isnan(position)):
                mean_markers[marker_name] += position
                valid_counts[marker_name] += 1

    # Mittlere Position für jeden Marker berechnen
    for marker_name in mean_markers:
        if valid_counts[marker_name] > 0:
            mean_markers[marker_name] /= valid_counts[marker_name]
        else:
            print(f"Warnung: Keine gültigen Positionen für Marker {marker_name} gefunden!")

    return mean_markers


def calculate_forceplate_corners_fixed(markers):
    """
    Berechnet die Eckpunkte für die Kraftplatten basierend auf den Markerpositionen.

    Verwendet die ursprünglichen gespeicherten Koordinaten direkt (ohne Transformation).

    WICHTIG: Die Mittelpunkte werden auf derselben Y-Höhe wie die äußeren Eckpunkte berechnet,
    um die leichte Versetzung der Kraftplatten zu berücksichtigen.
    """
    # Markerpositionen direkt verwenden (OHNE Koordinatentransformation)
    left_front = markers['LeftF']  # Vorne links
    right_front = markers['RightF']  # Vorne rechts
    left_back = markers['LeftB']  # Hinten links
    right_back = markers['RightB']  # Hinten rechts

    print("Verwendete Markerpositionen (original):")
    print(f"LeftF:  X={left_front[0]:.2f}, Y={left_front[1]:.2f}, Z={left_front[2]:.2f}")
    print(f"RightF: X={right_front[0]:.2f}, Y={right_front[1]:.2f}, Z={right_front[2]:.2f}")
    print(f"LeftB:  X={left_back[0]:.2f}, Y={left_back[1]:.2f}, Z={left_back[2]:.2f}")
    print(f"RightB: X={right_back[0]:.2f}, Y={right_back[1]:.2f}, Z={right_back[2]:.2f}")

    # KORRIGIERTE Berechnung der Mittelpunkte:
    # Die mittleren Eckpunkte sollen auf derselben Y-Höhe wie die äußeren Eckpunkte
    # der jeweiligen Kraftplatte liegen

    # Mittelpunkt vorne: X ist zwischen links und rechts, aber Y-Höhe bleibt gleich
    front_middle_fp4 = np.array([
        (left_front[0] + right_front[0]) / 2,  # X: Zwischen links und rechts
        left_front[1],  # Y: Gleiche Höhe wie LeftF
        (left_front[2] + right_front[2]) / 2  # Z: Durchschnitt
    ])

    front_middle_fp5 = np.array([
        (left_front[0] + right_front[0]) / 2,  # X: Zwischen links und rechts
        right_front[1],  # Y: Gleiche Höhe wie RightF
        (left_front[2] + right_front[2]) / 2  # Z: Durchschnitt
    ])

    # Mittelpunkt hinten: X ist zwischen links und rechts, aber Y-Höhe bleibt gleich
    back_middle_fp4 = np.array([
        (left_back[0] + right_back[0]) / 2,  # X: Zwischen links und rechts
        left_back[1],  # Y: Gleiche Höhe wie LeftB
        (left_back[2] + right_back[2]) / 2  # Z: Durchschnitt
    ])

    back_middle_fp5 = np.array([
        (left_back[0] + right_back[0]) / 2,  # X: Zwischen links und rechts
        right_back[1],  # Y: Gleiche Höhe wie RightB
        (left_back[2] + right_back[2]) / 2  # Z: Durchschnitt
    ])

    print(f"FP4 Front Middle: X={front_middle_fp4[0]:.2f}, Y={front_middle_fp4[1]:.2f}, Z={front_middle_fp4[2]:.2f}")
    print(f"FP4 Back Middle:  X={back_middle_fp4[0]:.2f}, Y={back_middle_fp4[1]:.2f}, Z={back_middle_fp4[2]:.2f}")
    print(f"FP5 Front Middle: X={front_middle_fp5[0]:.2f}, Y={front_middle_fp5[1]:.2f}, Z={front_middle_fp5[2]:.2f}")
    print(f"FP5 Back Middle:  X={back_middle_fp5[0]:.2f}, Y={back_middle_fp5[1]:.2f}, Z={back_middle_fp5[2]:.2f}")

    # Kraftplatte 4 (links) - mit korrigierten Mittelpunkten
    fp4_corners = np.array([
        left_front,  # Corner 1: Links vorne
        front_middle_fp4,  # Corner 2: Mitte vorne (auf Y-Höhe von LeftF)
        back_middle_fp4,  # Corner 3: Mitte hinten (auf Y-Höhe von LeftB)
        left_back  # Corner 4: Links hinten
    ])

    # Kraftplatte 5 (rechts) - mit korrigierten Mittelpunkten
    fp5_corners = np.array([
        front_middle_fp5,  # Corner 1: Mitte vorne (auf Y-Höhe von RightF)
        right_front,  # Corner 2: Rechts vorne
        right_back,  # Corner 3: Rechts hinten
        back_middle_fp5  # Corner 4: Mitte hinten (auf Y-Höhe von RightB)
    ])

    print("\nEckpunkt-Anordnung für FP4 (links):")
    print(f"Corner 1 (Links vorne): [{fp4_corners[0][0]:.2f}, {fp4_corners[0][1]:.2f}, {fp4_corners[0][2]:.2f}]")
    print(f"Corner 2 (Mitte vorne): [{fp4_corners[1][0]:.2f}, {fp4_corners[1][1]:.2f}, {fp4_corners[1][2]:.2f}]")
    print(f"Corner 3 (Mitte hinten): [{fp4_corners[2][0]:.2f}, {fp4_corners[2][1]:.2f}, {fp4_corners[2][2]:.2f}]")
    print(f"Corner 4 (Links hinten): [{fp4_corners[3][0]:.2f}, {fp4_corners[3][1]:.2f}, {fp4_corners[3][2]:.2f}]")

    print("\nEckpunkt-Anordnung für FP5 (rechts):")
    print(f"Corner 1 (Mitte vorne): [{fp5_corners[0][0]:.2f}, {fp5_corners[0][1]:.2f}, {fp5_corners[0][2]:.2f}]")
    print(f"Corner 2 (Rechts vorne): [{fp5_corners[1][0]:.2f}, {fp5_corners[1][1]:.2f}, {fp5_corners[1][2]:.2f}]")
    print(f"Corner 3 (Rechts hinten): [{fp5_corners[2][0]:.2f}, {fp5_corners[2][1]:.2f}, {fp5_corners[2][2]:.2f}]")
    print(f"Corner 4 (Mitte hinten): [{fp5_corners[3][0]:.2f}, {fp5_corners[3][1]:.2f}, {fp5_corners[3][2]:.2f}]")

    # Überprüfung der Y-Ausrichtung
    print(f"\nY-Ausrichtung Überprüfung:")
    print(f"FP4 vordere Linie: LeftF.Y={left_front[1]:.2f} = FP4_Middle_Front.Y={front_middle_fp4[1]:.2f}")
    print(f"FP4 hintere Linie: LeftB.Y={left_back[1]:.2f} = FP4_Middle_Back.Y={back_middle_fp4[1]:.2f}")
    print(f"FP5 vordere Linie: RightF.Y={right_front[1]:.2f} = FP5_Middle_Front.Y={front_middle_fp5[1]:.2f}")
    print(f"FP5 hintere Linie: RightB.Y={right_back[1]:.2f} = FP5_Middle_Back.Y={back_middle_fp5[1]:.2f}")

    return fp4_corners, fp5_corners


def update_forceplate_corners_in_c3d_fixed(input_c3d_path, output_c3d_path, new_corners_list):
    """
    Aktualisiert die Kraftplatten-Eckpunkte in einer C3D-Datei.
    Aktualisiert spezifisch Kraftplatte 4 und 5 (Indizes 3 und 4).
    Korrigiert auch COP-Ausrichtung durch Aktualisierung aller relevanten Parameter.
    """
    try:
        c3d = ezc3d.c3d(input_c3d_path)

        # Lesen der aktuellen Kraftplatten-Parameter
        current_corners = c3d['parameters']['FORCE_PLATFORM']['CORNERS']['value']
        current_origins = c3d['parameters']['FORCE_PLATFORM']['ORIGIN']['value']

        print(f"Aktuelle Corners Shape: {current_corners.shape}")
        print(f"Aktuelle Origins Shape: {current_origins.shape}")

        # Überprüfen, ob genügend Kraftplatten vorhanden sind
        n_plates = current_corners.shape[2]
        if n_plates < 5:
            print(f"Warnung: Nur {n_plates} Kraftplatten gefunden!")
            return

        # Kraftplatten-Indizes (0-basiert)
        fp4_index = 3  # Kraftplatte 4
        fp5_index = 4  # Kraftplatte 5

        print(f"Aktualisiere Kraftplatte 4 (Index {fp4_index}) und Kraftplatte 5 (Index {fp5_index})")

        # Neue Eckpunkte für Kraftplatten 4 und 5 setzen
        current_corners[:, :, fp4_index] = new_corners_list[0].T  # FP4 corners
        current_corners[:, :, fp5_index] = new_corners_list[1].T  # FP5 corners
        c3d['parameters']['FORCE_PLATFORM']['CORNERS']['value'] = current_corners

        # Berechne und aktualisiere die Ursprünge
        fp4_origin = np.mean(new_corners_list[0], axis=0)
        fp5_origin = np.mean(new_corners_list[1], axis=0)
        current_origins[:, fp4_index] = fp4_origin
        current_origins[:, fp5_index] = fp5_origin
        c3d['parameters']['FORCE_PLATFORM']['ORIGIN']['value'] = current_origins

        # Speichere die neue C3D-Datei
        c3d.write(output_c3d_path)
        print(f"\nNeue C3D-Datei gespeichert: {output_c3d_path}")
        print("Fallback-Methode verwendet - Grundlegende Geometrie aktualisiert!")

    except Exception as e:
        print(f"Fehler beim Aktualisieren der C3D-Datei: {e}")
        import traceback
        traceback.print_exc()


def visualize_results_fixed(markers, fp4_corners, fp5_corners):
    """
    Erstellt eine Visualisierung der Marker und der neuen Kraftplatteneckpunkte.
    Verwendet die ursprünglichen Koordinaten direkt.
    """
    fig = plt.figure(figsize=(15, 10))

    # 3D-Ansicht
    ax1 = fig.add_subplot(121, projection='3d')

    # Marker-Farben und Labels
    marker_colors = {
        'RightF': 'red',
        'RightB': 'darkred',
        'LeftF': 'blue',
        'LeftB': 'darkblue'
    }

    marker_labels = {
        'RightF': 'Right Front',
        'RightB': 'Right Back',
        'LeftF': 'Left Front',
        'LeftB': 'Left Back'
    }

    # Marker in 3D anzeigen (mit ursprünglichen Koordinaten)
    for marker_name, pos in markers.items():
        ax1.scatter(
            pos[0], pos[1], pos[2],  # X, Y, Z (ursprüngliche Koordinaten)
            color=marker_colors.get(marker_name, 'black'),
            s=100,
            label=marker_labels.get(marker_name, marker_name)
        )

        # Marker-Namen als Text hinzufügen
        ax1.text(pos[0], pos[1], pos[2], marker_name, fontsize=8)

    # Kraftplatten in 3D anzeigen
    plot_forceplate_3d_fixed(ax1, fp4_corners, 'FP4 (Links)', 'green')
    plot_forceplate_3d_fixed(ax1, fp5_corners, 'FP5 (Rechts)', 'orange')

    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_zlabel('Z [mm]')
    ax1.set_title('3D-Ansicht der Kraftplatten (Original-Koordinaten)')
    ax1.legend()

    # Draufsicht (X-Y-Ebene)
    ax2 = fig.add_subplot(122)

    # Marker in der Draufsicht (mit ursprünglichen Koordinaten)
    for marker_name, pos in markers.items():
        ax2.scatter(
            pos[0], pos[1],  # X, Y (ursprüngliche Koordinaten)
            color=marker_colors.get(marker_name, 'black'),
            s=100,
            label=marker_labels.get(marker_name, marker_name)
        )
        ax2.annotate(marker_name, (pos[0], pos[1]), xytext=(10, 10), textcoords='offset points')

    # Kraftplatten in der Draufsicht
    plot_forceplate_2d_fixed(ax2, fp4_corners, 'FP4 (Links)', 'green')
    plot_forceplate_2d_fixed(ax2, fp5_corners, 'FP5 (Rechts)', 'orange')

    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    ax2.set_title('Draufsicht der Kraftplatten (X-Y-Ebene)')
    ax2.grid(True)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig('kraftplatten_kalibrierung_final.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_forceplate_3d_fixed(ax, corners, label, color):
    """Zeichnet eine Kraftplatte als Viereck im 3D-Raum."""
    # Verbindungen zwischen den Eckpunkten (geschlossenes Viereck)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

    for i, j in lines:
        ax.plot(
            [corners[i, 0], corners[j, 0]],  # X
            [corners[i, 1], corners[j, 1]],  # Y
            [corners[i, 2], corners[j, 2]],  # Z
            color=color,
            linewidth=2
        )

    # Mittelpunkt und Label
    center = np.mean(corners, axis=0)
    ax.text(center[0], center[1], center[2], label, color=color, fontsize=10)


def plot_forceplate_2d_fixed(ax, corners, label, color):
    """Zeichnet eine Kraftplatte als Viereck in der X-Y-Ebene."""
    # Verbindungen zwischen den Eckpunkten
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

    for i, j in lines:
        ax.plot(
            [corners[i, 0], corners[j, 0]],  # X
            [corners[i, 1], corners[j, 1]],  # Y
            color=color,
            linewidth=2
        )

    # Eckpunkte nummerieren
    for i, corner in enumerate(corners):
        ax.text(corner[0], corner[1], str(i + 1), color='red', fontsize=8,
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', alpha=0.8))

    # Label am Mittelpunkt
    center = np.mean(corners, axis=0)
    ax.text(center[0], center[1], label, color=color, fontsize=10, ha='center')


if __name__ == "__main__":
    main()