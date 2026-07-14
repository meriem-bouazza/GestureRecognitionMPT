"""
Trainiert den HMMClassifier und speichert das Modell als data/hmm.pkl.

Ablauf:
    1. python record.py          # Gesten aufnehmen (A-Z, je 6x pro Person)
    2. python train.py           # <-- dieses Skript: trainiert & speichert data/hmm.pkl
    3. python main.py            # Live-Demo starten
"""
from GestureRecognition.hmmclassifier import HMMClassifier
from GestureRecognition.labeling import dataset_building

def main():
    # Alle Aufnahmen laden und Feature-Pipeline anwenden
    sequences, labels = dataset_building("data/recordings")

    if not sequences:
        raise FileNotFoundError(
            "Keine Aufnahmen gefunden. Erst Gesten aufnehmen:\n"
            "  python record.py"
        )

    print(f"Datensatz geladen: {len(set(labels))} Klassen, {len(sequences)} Sequenzen")

    # Training + Speichern
    clf = HMMClassifier(n_states=8, n_iter=100)
    clf.fit(sequences, labels)
    clf.save("data/hmm.pkl")
    print(f"Modell gespeichert: data/hmm.pkl  (Klassen: {', '.join(clf.classes)})")

if __name__ == "__main__":
    main()