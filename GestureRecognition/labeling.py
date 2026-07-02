import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import subprocess
import sys
import os


def data_labeling(times: int, label: str):
    """
    Datenerfassung für Gesten über SignalHub.

    Startet SignalHub im Record-Modus, nimmt Gesten auf und speichert
    sie nach Bereinigung strukturiert nach Label.

    Parameters
    ----------
    times : int
        Anzahl der zu speichernden Aufnahmen.
    label : str
        Name der Geste / Klasse (z. B. "A", "B").
    """
    temp_file = os.path.join("data", "temp_recording.pkl")
    saved = 0
    attempt = 0

    print(f"Starte Aufnahme-Session für Geste '{label}' ({times} Aufnahmen)")
    print("Steuerung: Geste vor Kamera ausführen, Fenster schließen.")
    print("           j = speichern, n = verwerfen, q = abbrechen\n")

    while saved < times:
        attempt += 1
        print(f"--- Aufnahme {saved + 1}/{times} (Versuch {attempt}) ---")

        result = subprocess.run(
            [sys.executable, "main.py",
             "--mode", "record",
             "--recorder.file", temp_file],
        )
        if result.returncode != 0:
            print("FEHLER beim Starten von SignalHub (siehe Ausgabe oben).")
            break

        if not os.path.exists(temp_file):
            print("Keine Aufnahme gefunden — erneut versuchen.\n")
            continue

        with open(temp_file, "rb") as f:
            rec = pickle.load(f)
        os.remove(temp_file)

        cleaned = clean_recording(rec)
        if cleaned is None:
            print("Keine Hand erkannt — Aufnahme verworfen.\n")
            continue

        choice = input("Speichern? (j/n/q): ").strip().lower()
        if choice == "q":
            print("Aufnahme-Session abgebrochen.")
            break
        elif choice == "j":
            path = save_recording(cleaned, label)
            saved += 1
            print(f"Gespeichert: {path} ({saved}/{times})\n")
        else:
            print("Aufnahme verworfen.\n")

    print(f"\nFertig! {saved} Aufnahmen für '{label}' gespeichert.")

def extract_trajectory(rec, finger_idx=8):
    trajectory = []
    for frame in rec["detector"]:
        result = frame["detector"]
        if not result.hand_landmarks:
            continue
        tip = result.hand_landmarks[0][finger_idx]
        trajectory.append((tip.x, tip.y))
    return trajectory


def normalize_trajectory(trajectory):
    traj = np.array(trajectory, dtype=np.float32)
    center = traj.mean(axis=0)
    traj -= center
    scale = max(traj[:, 0].max() - traj[:, 0].min(),
                traj[:, 1].max() - traj[:, 1].min())
    if scale > 0:
        traj /= scale
    return traj

def clean_recording(rec):
    """Schneidet Frames ohne erkannte Hand am Anfang und Ende weg."""
    frames = rec["detector"]

    # Für jeden Frame: war eine Hand drin? (True/False)
    has_hand = [bool(f["detector"].hand_landmarks) for f in frames if isinstance(f, dict) and "detector" in f]

    if not any(has_hand):
        return None  # gar keine Hand -> unbrauchbare Aufnahme

    start = has_hand.index(True)                              # erster Frame mit Hand
    end = len(has_hand) - 1 - has_hand[::-1].index(True)      # letzter Frame mit Hand

    cleaned_frames = frames[start:end + 1]
    return {"detector": cleaned_frames}


def save_recording(rec, label, base_dir="data/recordings"):
    """Extrahiert Trajektorie, normalisiert und speichert als .npy (T, 2) float32."""
    trajectory = extract_trajectory(rec)
    if len(trajectory) < 2:
        return None

    traj = normalize_trajectory(trajectory)

    label_dir = Path(base_dir) / label
    label_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = label_dir / f"{timestamp}.npy"
    np.save(path, traj)

    return path
    

def dataset_building(output_path):
    """
    TODO: dataset_building: Trainingsdatensatz aus aufgenommenen Gesten erstellen

    Ziel:
    -----
    Implementiere eine Funktion, die alle aufgenommenen Daten lädt,
    verarbeitet und in eine Form bringt, die von eurem
    Hidden-Markov-Modell (HMM) Classifier verwendet werden kann.

    Anforderungen / Ideen:
    ----------------------

    1. Daten laden

       - Durchsuche deinen Trainingsdaten-Ordner
       - Organisiere Daten nach Labels

    2. Feature-Extraktion / Preprocessing

       - Überlege:
         - Welche Features braucht dein Modell?
         - Wie transformierst du die Rohdaten sinnvoll?
       - Wende eine konsistente Verarbeitung auf alle Sequenzen an

    3. Umgang mit Sequenzen

       - Daten sind zeitliche Sequenzen
       - Achte auf:
         - Unterschiedliche Längen
         - Konsistente Struktur

    4. Validierung

       - Entferne unbrauchbare Daten
         (z. B. zu kurze oder fehlerhafte Sequenzen)

    5. Ausgabeformat

       - Baue den Datensatz so, dass dein HMM direkt damit arbeiten kann
       - Das Format sollst du selbst definieren

    .. note::

       Es gibt hier keine vorgegebene „richtige" Lösung.
       Wichtig ist, dass dein Datensatz konsistent und nutzbar ist.

    .. tip::

       Denke wie ein System-Designer:
       Wie müssen Daten aussehen, damit Training und Inferenz sauber funktionieren?

    .. warning::

       Inkonsistente Datenstrukturen sind eine der häufigsten Fehlerquellen
       beim Training von Sequenzmodellen.

    Erweiterung (optional):
    -----------------------

    - Normalisierung der Daten
    - Datenaugmentation
    - Debug-Ausgaben oder Visualisierung

    Parameters
    ----------
    output_path : Path or str
        Zielpfad für den erzeugten Trainingsdatensatz.
    """
    
    recordings_dir = Path("data/recordings")

    sequences = []
    labels = []

    for npy_file in sorted(recordings_dir.glob("*/*.npy")):
        label = npy_file.parent.name
        traj = np.load(npy_file)

        if traj.shape[0] < 2:
            continue

        sequences.append(traj)
        labels.append(label)

    print(f"{len(sequences)} Sequenzen geladen, Klassen: {sorted(set(labels))}")
    return sequences, labels