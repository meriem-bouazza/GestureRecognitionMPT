import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GestureRecognition"))

from labeling import data_labeling

GESTURES = ["H", "W", "X"]
PER_PERSON = 6
PEOPLE = 5
TARGET_TOTAL = PER_PERSON * PEOPLE  # 30
RECORDINGS_DIR = Path("data/recordings")

print("Jede Person nimmt 6 Aufnahmen pro Geste auf (5x6=30 neue pro Geste).\n")
print(f"Gesten: {', '.join(GESTURES)}  ({len(GESTURES)} Stueck)\n")

for gesture in GESTURES:
    existing = len(list((RECORDINGS_DIR / gesture).glob("*.npy")))
    if existing >= TARGET_TOTAL:
        print(f"Geste {gesture}: bereits {existing}/{TARGET_TOTAL} vorhanden, übersprungen.")
        continue
    remaining = min(PER_PERSON, TARGET_TOTAL - existing)

    print(f"\n{'='*40}")
    print(f"  Geste: {gesture}  --  {remaining} Aufnahmen (bereits {existing}/{TARGET_TOTAL})")
    print(f"{'='*40}")
    data_labeling(times=remaining, label=gesture)

print("\nFertig! Naechste Person kann starten.")
