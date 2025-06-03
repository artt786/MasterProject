import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d


def load_mot_file(file_path):
    """Lädt MOT-Datei und extrahiert Zeit und ankle_angle_r"""
    print(f"\nLading file: {os.path.basename(file_path)}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Debug: Zeige erste paar Zeilen
    print(f"First 10 lines of file:")
    for i, line in enumerate(lines[:10]):
        print(f"Line {i}: {repr(line)}")

    # Finde Header-Ende - robustere Methode
    header_end = 0
    data_start = 0

    # Suche nach 'endheader'
    for i, line in enumerate(lines):
        if 'endheader' in line.lower():
            header_end = i + 1
            data_start = i + 2  # Headers sind in der nächsten Zeile
            break
    else:
        # Kein 'endheader' gefunden, suche nach der ersten Zeile mit Zahlen
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith(('#', 'version', 'nRows', 'nColumns', 'inDegrees', 'name')):
                # Prüfe ob diese Zeile Headers oder Daten enthält
                parts = line.split()
                try:
                    # Wenn alle Teile Zahlen sind, ist das die Datenzeile
                    [float(x) for x in parts]
                    # Das sind Daten, also ist die vorherige Zeile der Header
                    if i > 0:
                        header_end = i - 1
                        data_start = i
                    else:
                        # Erste Zeile sind schon Daten, erstelle synthetische Headers
                        print("Warning: No headers found, creating synthetic headers")
                        headers = [f"col_{j}" for j in range(len(parts))]
                        headers[0] = "time"  # Erste Spalte als Zeit annehmen
                        header_end = -1  # Flag für synthetische Headers
                        data_start = 0
                    break
                except ValueError:
                    # Das sind Headers (enthalten Text)
                    header_end = i
                    data_start = i + 1
                    break

    # Lade Headers
    if header_end >= 0:
        headers = lines[header_end].strip().split()
    # headers wurde bereits oben definiert für synthetische Headers

    print(f"Headers found: {headers}")
    print(f"Header line index: {header_end}")
    print(f"Data starts at line: {data_start}")

    # Finde Spaltenindices
    if 'time' not in headers:
        raise ValueError(f"'time' column not found. Available columns: {headers}")

    time_idx = headers.index('time')

    # Suche ankle_angle_r Spalte
    ankle_idx = None
    possible_ankle_names = ['ankle_angle_r', 'ankle_angle_right', 'ankle_r', 'AnkleAngle_r']
    for ankle_name in possible_ankle_names:
        if ankle_name in headers:
            ankle_idx = headers.index(ankle_name)
            break

    if ankle_idx is None:
        raise ValueError(f"ankle_angle_r column not found. Available columns: {headers}")

    print(f"Time column index: {time_idx}")
    print(f"Ankle angle column index: {ankle_idx}")

    # Lade Daten
    data = []
    for i, line in enumerate(lines[data_start:], start=data_start):
        line = line.strip()
        if line and not line.startswith('#'):
            try:
                values = [float(x) for x in line.split()]
                if len(values) == len(headers):
                    data.append(values)
                else:
                    print(f"Warning: Line {i} has {len(values)} values, expected {len(headers)}")
            except ValueError as e:
                print(f"Warning: Could not parse line {i}: {line[:50]}... Error: {e}")
                continue

    if not data:
        raise ValueError("No valid data found in file")

    data = np.array(data)
    print(f"Data shape: {data.shape}")

    time = data[:, time_idx]
    ankle_angle = data[:, ankle_idx]

    print(f"Time range: {time[0]:.3f} - {time[-1]:.3f} seconds")
    print(f"Ankle angle range: {ankle_angle.min():.1f} - {ankle_angle.max():.1f} degrees")

    return time, ankle_angle


def calculate_angular_velocity(time, angle, smooth_sigma=1.0):
    """
    Berechnet die Winkelgeschwindigkeit mit zentraler Differenzenformel:
    ω_i ≈ (θ_{i+1} - θ_{i-1}) / (t_{i+1} - t_{i-1})

    Parameters:
    - time: Zeitarray
    - angle: Winkelarray in Grad
    - smooth_sigma: Sigma für Gaussian-Glättung (0 = keine Glättung)

    Returns:
    - angular_velocity: Winkelgeschwindigkeit in °/s
    """
    # Glättung des Winkelsignals vor Differentiation (optional)
    if smooth_sigma > 0:
        angle_smooth = gaussian_filter1d(angle, sigma=smooth_sigma)
    else:
        angle_smooth = angle.copy()

    # Initialisiere Winkelgeschwindigkeit Array
    angular_velocity = np.zeros_like(time)

    # Zentrale Differenzenformel für innere Punkte: ω_i ≈ (θ_{i+1} - θ_{i-1}) / (t_{i+1} - t_{i-1})
    for i in range(1, len(time) - 1):
        angular_velocity[i] = (angle_smooth[i + 1] - angle_smooth[i - 1]) / (time[i + 1] - time[i - 1])

    # Randpunkte: Vorwärts- und Rückwärts-Differenz
    # Erster Punkt: Vorwärts-Differenz
    angular_velocity[0] = (angle_smooth[1] - angle_smooth[0]) / (time[1] - time[0])

    # Letzter Punkt: Rückwärts-Differenz
    angular_velocity[-1] = (angle_smooth[-1] - angle_smooth[-2]) / (time[-1] - time[-2])

    return angular_velocity


def save_to_excel(ankle_data, base_path):
    """
    Speichert die Daten für jeden Schritt in eine separate Excel-Datei
    """
    excel_dir = os.path.join(base_path, "Excel_Results")

    # Erstelle Ordner falls nicht vorhanden
    if not os.path.exists(excel_dir):
        os.makedirs(excel_dir)
        print(f"Created directory: {excel_dir}")

    for step, data in ankle_data.items():
        # Erstelle DataFrame
        df = pd.DataFrame({
            'Time_s': data['time'],
            'Ankle_Angle_deg': data['ankle_angle'],
            'Angular_Velocity_deg_per_s': data['angular_velocity'],
            'Stance_Phase_percent': np.linspace(0, 100, len(data['time']))
        })

        # Berechne Statistiken
        stats_df = pd.DataFrame({
            'Parameter': ['Mean', 'Std', 'Min', 'Max', 'Range'],
            'Ankle_Angle_deg': [
                data['ankle_angle'].mean(),
                data['ankle_angle'].std(),
                data['ankle_angle'].min(),
                data['ankle_angle'].max(),
                data['ankle_angle'].max() - data['ankle_angle'].min()
            ],
            'Angular_Velocity_deg_per_s': [
                data['angular_velocity'].mean(),
                data['angular_velocity'].std(),
                data['angular_velocity'].min(),
                data['angular_velocity'].max(),
                data['angular_velocity'].max() - data['angular_velocity'].min()
            ]
        })

        # Speichere in Excel mit mehreren Sheets
        excel_filename = os.path.join(excel_dir, f"Step_{step}_ankle_analysis.xlsx")

        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        print(f"Saved: {excel_filename}")

    # Erstelle auch eine Zusammenfassung aller Schritte
    summary_data = []
    for step, data in ankle_data.items():
        summary_data.append({
            'Step': step,
            'Duration_s': data['time'][-1] - data['time'][0],
            'Samples': len(data['time']),
            'Ankle_Angle_Mean_deg': data['ankle_angle'].mean(),
            'Ankle_Angle_Std_deg': data['ankle_angle'].std(),
            'Ankle_Angle_Range_deg': data['ankle_angle'].max() - data['ankle_angle'].min(),
            'Angular_Velocity_Mean_deg_per_s': data['angular_velocity'].mean(),
            'Angular_Velocity_Std_deg_per_s': data['angular_velocity'].std(),
            'Angular_Velocity_Max_deg_per_s': data['angular_velocity'].max(),
            'Angular_Velocity_Min_deg_per_s': data['angular_velocity'].min()
        })

    summary_df = pd.DataFrame(summary_data)
    summary_filename = os.path.join(excel_dir, "All_Steps_Summary.xlsx")
    summary_df.to_excel(summary_filename, index=False)
    print(f"Saved summary: {summary_filename}")

    return excel_dir


def visualize_ankle_angles():
    """Hauptfunktion - lädt alle MOT-Dateien und visualisiert ankle_angle_r"""

    # Pfad zu IK-Ergebnissen
    base_path = r"C:\Users\annar\OneDrive - The University of Auckland\Master\ENGGEN 790\Pilot"
    ik_results_dir = os.path.join(base_path, "IK_Results")

    # Überprüfe ob Ordner existiert
    if not os.path.exists(ik_results_dir):
        print(f"Error: Directory {ik_results_dir} not found!")
        return

    # Finde MOT-Dateien
    mot_files = []
    for file in os.listdir(ik_results_dir):
        if file.endswith('.mot') and 'step_' in file:
            mot_files.append(os.path.join(ik_results_dir, file))

    mot_files.sort()
    print(f"Gefunden: {len(mot_files)} MOT-Dateien")

    if len(mot_files) == 0:
        print("No MOT files found!")
        return

    # Lade Daten
    ankle_data = {}
    successful_loads = 0

    for i, mot_file in enumerate(mot_files):
        try:
            print(f"\nProcessing file: {os.path.basename(mot_file)}")
            time, ankle_angle = load_mot_file(mot_file)

            # Berechne Winkelgeschwindigkeit
            angular_velocity = calculate_angular_velocity(time, ankle_angle, smooth_sigma=1.0)

            ankle_data[i + 1] = {
                'time': time,
                'ankle_angle': ankle_angle,
                'angular_velocity': angular_velocity
            }
            print(f"Schritt {i + 1}: {len(time)} Frames, {time[-1] - time[0]:.3f}s")
            print(f"  Ankle angle range: {ankle_angle.min():.1f}° to {ankle_angle.max():.1f}°")
            print(f"  Angular velocity range: {angular_velocity.min():.1f}°/s to {angular_velocity.max():.1f}°/s")
            successful_loads += 1
        except Exception as e:
            print(f"Error loading {mot_file}: {e}")
            continue

    if successful_loads == 0:
        print("No files could be loaded successfully!")
        return

    # Speichere Daten in Excel-Dateien
    print(f"\nSaving data to Excel files...")
    excel_dir = save_to_excel(ankle_data, base_path)

    # Visualisierung
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


    # Plot 5: Ankle Angles normalisiert auf % Stance Phase
    plt.figure(figsize=(12, 6))
    for step, data in ankle_data.items():
        stance_percent = np.linspace(0, 100, len(data['ankle_angle']))
        color_idx = (step - 1) % len(colors)
        plt.plot(stance_percent, data['ankle_angle'],
                 color=colors[color_idx], linewidth=2, label=f'Schritt {step}', alpha=0.8)

    plt.title('Ankle Angle - Alle Schritte (% Stance Phase)')
    plt.xlabel('Stance Phase [%]')
    plt.ylabel('Ankle Angle [°]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.show()

    # Plot 6: Angular Velocities normalisiert auf % Stance Phase
    plt.figure(figsize=(12, 6))
    for step, data in ankle_data.items():
        stance_percent = np.linspace(0, 100, len(data['angular_velocity']))
        color_idx = (step - 1) % len(colors)
        plt.plot(stance_percent, data['angular_velocity'],
                 color=colors[color_idx], linewidth=2, label=f'Schritt {step}', alpha=0.8)

    plt.title('Angular Velocity - Alle Schritte (% Stance Phase)')
    plt.xlabel('Stance Phase [%]')
    plt.ylabel('Angular Velocity [°/s]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.show()



    # Statistics
    print(f"\n" + "=" * 80)
    print(f"ZUSAMMENFASSUNG")
    print(f"=" * 80)
    print(f"Successfully loaded {successful_loads} files")
    print(f"Excel files saved to: {excel_dir}")
    print(f"\nStatistiken für jeden Schritt:")

    for step, data in ankle_data.items():
        print(f"\nSchritt {step}:")
        print(f"  Duration: {data['time'][-1] - data['time'][0]:.3f}s")
        print(f"  Ankle Angle - Min: {np.min(data['ankle_angle']):.1f}°, "
              f"Max: {np.max(data['ankle_angle']):.1f}°, "
              f"Mean: {np.mean(data['ankle_angle']):.1f}°, "
              f"Std: {np.std(data['ankle_angle']):.1f}°")
        print(f"  Angular Velocity - Min: {np.min(data['angular_velocity']):.1f}°/s, "
              f"Max: {np.max(data['angular_velocity']):.1f}°/s, "
              f"Mean: {np.mean(data['angular_velocity']):.1f}°/s, "
              f"Std: {np.std(data['angular_velocity']):.1f}°/s")


if __name__ == "__main__":
    visualize_ankle_angles()