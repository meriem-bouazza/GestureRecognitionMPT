import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GestureRecognition"))

from labeling import data_labeling

# melin: Einzelausreißer ersetzen
TO_RECORD = {
    "B": 1,
    "V": 1,
    "D": 1,
    "G": 1,
    "Z": 2,
    "Q": 1,
    "S": 1,
    "C": 1,
}

print("Ausreißer-Session für melin")
print("Folgende Aufnahmen werden gemacht:")
for letter, count in TO_RECORD.items():
    print(f"  {letter}: {count}x")
print()

for letter, count in TO_RECORD.items():
    print(f"\n{'='*40}")
    print(f"  Geste: {letter}  --  {count} Aufnahme(n)")
    print(f"{'='*40}")
    data_labeling(times=count, label=letter)

print("\nFertig! Alle Ausreißer ersetzt.")
