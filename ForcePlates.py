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
    output_path = r'C:\Users\annar\OneDrive - The University of Auckland\Master\ENGGEN 790\Treadmill\New Session\CalibratedForcePlates'

    # Erstellen des Ausgabeordners, falls er nicht existiert
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Einlesen aller C3D-Dateien
    c3d_files = [f for f in os.listdir(c3d_path) if f.startswith('LeftFP_') and f.endswith('.c3d')]
    c3d_files.sort()  # Sortieren, um sicherzustellen, dass die Dateien in der richtigen Reihenfolge verarbeitet werden

    print(f"Gefundene C3D-Dateien: {len(c3d_files)}")


    # Sammeln der Markerpositionen aus allen Dateien
    marker_positions = {}

    for c3d_file in c3d_files:
        file_base = c3d_file.replace('.c3d', '')
        print(f"Verarbeite Datei: {c3d_file}")

        try:
            # Laden der C3D-Datei
            mocap = MocapDO.create_from_c3d(os.path.join(c3d_path, c3d_file))

            # Umwandeln in Y-Up-Koordinatensystem
            mocap.markers.z_up_to_y_up()

            # Speichern der Markerpositionen
            marker_positions[file_base] = {}
            for marker_name in ['*6', '*7', '*8', '*9']:
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

    # Korrigierte Funktion zur Berechnung der Kraftplattenposition
    # Basierend auf dem Bild erwarten wir zwei Rechtecke nebeneinander
    # Marker 6 (oben links), Marker 7 (oben rechts), Marker 8 (unten links), Marker 9 (unten rechts)
    new_fp4_corners, new_fp5_corners = calculate_forceplate_corners_corrected(mean_markers)

    for c3d_file in c3d_files:
        input_c3d = os.path.join(c3d_path, c3d_file)
        output_c3d = os.path.join(output_path, f'Calibrated_{c3d_file}')
        update_forceplate_corners_in_c3d(input_c3d, output_c3d, [new_fp4_corners, new_fp5_corners])

    print("Neue Kraftplatteneckpunkte berechnet:")
    print("FP4 Eckpunkte (links):", new_fp4_corners)
    print("FP5 Eckpunkte (rechts):", new_fp5_corners)

    # Kraftplattenindizes festlegen
    fp4_index = 3  # 0-basierter Index für Kraftplatte 4
    fp5_index = 4  # 0-basierter Index für Kraftplatte 5

    print(f"Verwende Index {fp4_index} für Kraftplatte 4 (links)")
    print(f"Verwende Index {fp5_index} für Kraftplatte 5 (rechts)")



    # Visualisieren der Ergebnisse
    visualize_results(mean_markers, new_fp4_corners, new_fp5_corners)

    print("Kalibrierung abgeschlossen!")
    print("\nBitte beachten Sie: Da die automatische Aktualisierung der Kraftplattenposition in den C3D-Dateien")
    print("nicht möglich war, wurden Konfigurationsdateien erstellt, die die neuen Koordinaten enthalten.")
    print("Diese können in Vicon Nexus verwendet werden, um die Kraftplatten manuell zu positionieren.")


def calculate_mean_markers(marker_positions):
    """Berechnet die mittleren Positionen aller Marker über alle gültigen Dateien."""
    mean_markers = {}
    valid_counts = {}

    for marker_name in ['*6', '*7', '*8', '*9']:
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


def calculate_forceplate_corners_corrected(markers):
    """
    Berechnet die Eckpunkte für die Kraftplatten basierend auf den Markerpositionen.

    Forceplate 4 (links):
        Corner 1: Marker 6 (oben links)
        Corner 2: Mitte oben (zwischen Marker 6 und 7)
        Corner 3: Marker 8 (unten links)
        Corner 4: Mitte unten (zwischen Marker 8 und 9)

    Forceplate 5 (rechts):
        Corner 1: Mitte oben (zwischen Marker 6 und 7)
        Corner 2: Marker 7 (oben rechts)
        Corner 3: Mitte unten (zwischen Marker 8 und 9)
        Corner 4: Marker 9 (unten rechts)
    """
    # Markerpositionen extrahieren
    top_left = markers['*6']      # Marker 6
    top_right = markers['*7']     # Marker 7
    bottom_left = markers['*8']   # Marker 8
    bottom_right = markers['*9']  # Marker 9

    # Mittelpunkte berechnen
    top_middle = (top_left + top_right) / 2
    bottom_middle = (bottom_left + bottom_right) / 2

    # Eckpunkte zuordnen
    fp4_corners = np.array([top_left, top_middle, bottom_left, bottom_middle])
    fp5_corners = np.array([top_middle, top_right, bottom_middle, bottom_right])

    return fp4_corners, fp5_corners

def update_forceplate_corners_in_c3d(input_c3d_path, output_c3d_path, new_corners_list):
    """
    new_corners_list: Liste von 4x3 numpy arrays, je Kraftplatte (z.B. [fp4_corners, fp5_corners])
    """
    c3d = ezc3d.c3d(input_c3d_path)
    # Die Ecken müssen shape (3,4,2) haben: (X,Y,Z), 4 Ecken, 2 Platten
    new_corners = np.stack([corners.T for corners in new_corners_list], axis=2)  # shape (3,4,2)
    c3d['parameters']['FORCE_PLATFORM']['CORNERS']['value'] = new_corners

    # Optional: Origin anpassen (Mittelpunkt der Platte)
    new_origins = np.stack([np.mean(corners, axis=0) for corners in new_corners_list], axis=1)  # shape (3,2)
    c3d['parameters']['FORCE_PLATFORM']['ORIGIN']['value'] = new_origins

    c3d.write(output_c3d_path)
    print(f"Neue C3D-Datei gespeichert: {output_c3d_path}")




def visualize_results(markers, fp4_corners, fp5_corners):
    """
    Erstellt eine Visualisierung der Marker und der neuen Kraftplatteneckpunkte.
    Zeigt eine Draufsicht (X-Z-Ebene) und eine 3D-Ansicht.
    """
    # Erstellen einer 3D-Figur
    fig = plt.figure(figsize=(15, 10))

    # Erstellen einer 3D-Visualisierung
    ax1 = fig.add_subplot(121, projection='3d')

    # Marker anzeigen
    marker_colors = {
        '*6': 'red',
        '*7': 'green',
        '*8': 'blue',
        '*9': 'purple'
    }

    marker_labels = {
        '*6': 'Marker 6 (oben links)',
        '*7': 'Marker 7 (oben rechts)',
        '*8': 'Marker 8 (unten links)',
        '*9': 'Marker 9 (unten rechts)'
    }

    for marker_name, pos in markers.items():
        ax1.scatter(
            pos[0], pos[2], pos[1],  # X, Z, Y
            color=marker_colors.get(marker_name, 'black'),
            s=100,
            label=marker_labels.get(marker_name, marker_name)
        )

    # Kraftplatten in 3D anzeigen
    plot_forceplate_3d(ax1, fp4_corners, 'Kraftplatte 4 (links)', 'blue')
    plot_forceplate_3d(ax1, fp5_corners, 'Kraftplatte 5 (rechts)', 'green')


    # Achsenbeschriftungen
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Z [mm]')
    ax1.set_zlabel('Y [mm]')

    # Titel
    ax1.set_title('3D-Ansicht der Kraftplatten')

    # Legende
    ax1.legend()

    # Achsengleiche Skalierung für bessere Darstellung in 3D
    max_range = np.array([
        ax1.get_xlim()[1] - ax1.get_xlim()[0],
        ax1.get_ylim()[1] - ax1.get_ylim()[0],
        ax1.get_zlim()[1] - ax1.get_zlim()[0]
    ]).max() / 2.0

    mid_x = (ax1.get_xlim()[1] + ax1.get_xlim()[0]) / 2
    mid_y = (ax1.get_ylim()[1] + ax1.get_ylim()[0]) / 2
    mid_z = (ax1.get_zlim()[1] + ax1.get_zlim()[0]) / 2

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # Draufsicht (X-Z-Ebene)
    ax2 = fig.add_subplot(122)

    # Marker in der Draufsicht anzeigen
    for marker_name, pos in markers.items():
        ax2.scatter(
            pos[0], pos[1],  # X, Y
            color=marker_colors.get(marker_name, 'black'),
            s=100,
            label=marker_labels.get(marker_name, marker_name)
        )
        ax2.annotate(marker_name, (pos[0], pos[1]), xytext=(10, 10), textcoords='offset points')

    # Kraftplatten in der Draufsicht anzeigen (X-Y)
    plot_forceplate_2d_xy(ax2, fp4_corners, 'Kraftplatte 4 (links)', 'blue')
    plot_forceplate_2d_xy(ax2, fp5_corners, 'Kraftplatte 5 (rechts)', 'green')



    # Achsenbeschriftungen
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')

    # Titel
    ax2.set_title('Draufsicht der Kraftplatten (X-Y-Ebene)')
    ax2.grid(True)
    ax2.axis('equal')  # Gleiche Skalierung für X und Z

    plt.tight_layout()

    # Speichern und Anzeigen
    plt.savefig('kraftplatten_kalibrierung.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_forceplate_3d(ax, corners, label, color):
    """Zeichnet eine Kraftplatte als Viereck im 3D-Raum."""
    # Verbindungen zwischen den Eckpunkten
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0]  # Umriss der Kraftplatte
    ]

    # Zeichnen der Linien
    for i, j in lines:
        ax.plot(
            [corners[i, 0], corners[j, 0]],  # X-Koordinaten
            [corners[i, 2], corners[j, 2]],  # Z-Koordinaten
            [corners[i, 1], corners[j, 1]],  # Y-Koordinaten
            color=color,
            linewidth=2
        )

    # Mittelpunkt berechnen
    center = np.mean(corners, axis=0)

    # Beschriftung am Mittelpunkt
    ax.text(
        center[0], center[2], center[1],
        label,
        color=color,
        fontsize=10,
        horizontalalignment='center'
    )

def plot_forceplate_2d_xy(ax, corners, label, color):
        """Zeichnet eine Kraftplatte als Viereck in der X-Y-Ebene (Draufsicht)."""
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0]
        ]
        for i, j in lines:
            ax.plot(
                [corners[i, 0], corners[j, 0]],  # X-Koordinaten
                [corners[i, 1], corners[j, 1]],  # Y-Koordinaten
                color=color,
                linewidth=2
            )
        center = np.mean(corners, axis=0)
        ax.text(
            center[0], center[1],
            label,
            color=color,
            fontsize=10,
            horizontalalignment='center'
        )




if __name__ == "__main__":
    main()