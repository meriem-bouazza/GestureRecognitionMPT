"""
Live-Aufnahme von Gesten-Trajektorien mit direkter Steuerung im Kamerafenster.

Anders als labeling.data_labeling (SignalHub-Subprozess + Terminal-Eingabe)
laeuft hier alles in einem OpenCV-Fenster, sodass man waehrend des Filmens
per Tastendruck speichern / verwerfen / abbrechen kann. Ausserdem startet die
Aufnahme erst auf Tastendruck - so hat man Zeit, den Finger zu positionieren.

Steuerung (Tasten im Fenster):
    Leertaste : Aufnahme starten / neu beginnen (Puffer leeren)
    S         : aktuelle Spur als .npy speichern
    D         : aktuelle Spur verwerfen
    Q / ESC   : beenden

Aufruf:
    python record_live.py [BUCHSTABE]      # Standard: A

Speicherort (gleiches Format wie labeling.save_recording):
    data/recordings/<BUCHSTABE>/<timestamp>.npy   -> Shape (T, 2), float32, normalisiert
"""
import sys
import os
import urllib.request
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

FINGER_IDX = 8          # MediaPipe-Landmark: Zeigefinger-Tip
DEVICE_INDEX = 0        # Webcam-Index (siehe config.yml webcam.deviceIndex)
MIN_POINTS = 15         # darunter wird nicht gespeichert (vgl. preprocessor.min_steps)
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def load_landmarker():
    if not os.path.exists(MODEL_PATH):
        print("Lade MediaPipe-Modell ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def normalize_trajectory(points):
    """Zentriert auf Schwerpunkt, skaliert auf max. Bounding-Box-Seite."""
    traj = np.array(points, dtype=np.float32)
    center = traj.mean(axis=0)
    traj -= center
    scale = max(traj[:, 0].max() - traj[:, 0].min(),
                traj[:, 1].max() - traj[:, 1].min())
    if scale > 0:
        traj /= scale
    return traj


def save_trajectory(points, label, base_dir="data/recordings"):
    if len(points) < MIN_POINTS:
        return None
    traj = normalize_trajectory(points)
    label_dir = os.path.join(base_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(label_dir, f"{timestamp}.npy")
    np.save(path, traj)
    return path


def main():
    label = sys.argv[1].upper() if len(sys.argv) > 1 else "A"

    landmarker = load_landmarker()
    cap = cv2.VideoCapture(DEVICE_INDEX)
    if not cap.isOpened():
        print(f"Kamera (Index {DEVICE_INDEX}) konnte nicht geoeffnet werden. "
              "Andere App (z.B. Teams) schliessen oder deviceIndex pruefen.")
        return

    recording = False
    points = []                  # rohe (x, y) fuers Speichern
    trail = deque(maxlen=300)    # Pixelpunkte fuer die Anzeige
    saved_count = 0
    message = ""

    print(f"Aufnahme fuer Geste '{label}'.  "
          "Leertaste=Start  S=Speichern  D=Verwerfen  Q=Beenden")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            tip = result.hand_landmarks[0][FINGER_IDX]
            px, py = int(tip.x * w), int(tip.y * h)
            cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)
            if recording:
                points.append((tip.x, tip.y))
                trail.append((px, py))

        # Spur zeichnen
        pts = list(trail)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 3)

        # Status
        status = "AUFNAHME..." if recording else "BEREIT"
        color = (0, 0, 255) if recording else (0, 200, 0)
        cv2.putText(frame, f"Geste {label}  |  {status}  |  gespeichert: {saved_count}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "Leertaste=Start  S=Speichern  D=Verwerfen  Q=Beenden",
                    (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if message:
            cv2.putText(frame, message, (15, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Gesten-Aufnahme", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            recording = True
            points = []
            trail.clear()
            message = "Aufnahme laeuft - zeichne den Buchstaben"
        elif key == ord('s'):
            if recording:
                path = save_trajectory(points, label)
                if path:
                    saved_count += 1
                    message = f"Gespeichert: {os.path.basename(path)}"
                else:
                    message = f"Zu kurz (<{MIN_POINTS} Punkte) - nicht gespeichert"
                recording = False
                points = []
                trail.clear()
        elif key == ord('d'):
            recording = False
            points = []
            trail.clear()
            message = "Verworfen"
        elif key == ord('q') or key == 27:  # q oder ESC
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print(f"Beendet. {saved_count} Aufnahmen fuer '{label}' gespeichert.")


if __name__ == "__main__":
    main()
