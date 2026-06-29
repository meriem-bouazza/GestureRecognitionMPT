import sys
sys.path.append("GestureRecognition")

import numpy as np
from hmmclassifier import HMMClassifier

classifier = HMMClassifier(n_states=3, n_iter=5)
sequences = [np.random.randn(30, 2) for _ in range(6)]
labels = ["A", "A", "A", "B", "B", "B"]
classifier.fit(sequences, labels)

# speichern
classifier.save("tests/test_model.pkl")

# laden
loaded = HMMClassifier.load("tests/test_model.pkl")

# prüfen
print("Klassen:", loaded.classes)       # ['A', 'B']
print("Modelle:", list(loaded.models.keys()))  # ['A', 'B']
print("Vorhersage:", loaded.predict(np.random.randn(30, 2)))  # 'A' oder 'B'

# Funktioniert, save und load korrekt