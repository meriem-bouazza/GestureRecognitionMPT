import pickle
import numpy as np
from hmmlearn.hmm import GaussianHMM

class HMMClassifier:
    """
    TODO: Implementiere einen HMM-basierten Klassifikator

    Ziel:
    -----
    Entwickle einen Klassifikator, der zeitliche Sequenzen mit Hilfe von
    Hidden-Markov-Modellen (HMMs) klassifiziert. Für HMMs können libraries wie
    :mod:`hmmlearn` benutzt werden

    Grundidee:
    ----------
    - Trainiere ein Modell pro Klasse
    - Bewerte neue Sequenzen anhand der Likelihood unter jedem Modell
    - Wähle die Klasse mit der höchsten Wahrscheinlichkeit

    .. note::
       Wie genau deine Modelle aussehen (z. B. Anzahl Zustände, Features,
       Initialisierung etc.) ist bewusst nicht vorgegeben.

    Wichtige Designentscheidungen:
    ------------------------------
    - Wie strukturierst du deine Trainingsdaten?
    - Wie repräsentierst du Sequenzen?
    - Wie verbindest du mehrere Sequenzen mit Labels?

    Speicherung:
    ------------
    Du solltest dir überlegen:
    - Wie speicherst du dein trainiertes Modell?
    - Wie lädst du es später wieder?
    - Welche Informationen müssen persistiert werden (z. B. Klassen, Modelle)?

    .. tip::
       ``pickle`` ist eine einfache Möglichkeit, Modelle zu speichern.
       Alternativ kannst du auch eigene Formate definieren.

    Evaluation:
    -----------
    Für sinnvolles Training solltest du unbedingt:
    - eine eigene ``train_test_split``-Logik implementieren
    - Trainings- und Testdaten sauber trennen

    .. warning::
       Wenn du Training und Test nicht trennst, sind deine Ergebnisse nicht aussagekräftig.

    Erweiterung (optional):
    -----------------------
    - Implementiere eine Grid Search für Hyperparameter
      (z. B. Anzahl Zustände, Modellstruktur)
    - Vergleiche verschiedene Modellkonfigurationen

    """

    def __init__(self, n_states: int = 5, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter = n_iter
        self.models = {}
        self.classes = []

    def fit(self, sequences: list, labels: list):
        """Trainiert ein GaussianHMM pro Klasse mit hmmlearn."""
        sequences = [np.array(seq) for seq in sequences]
        self.classes = list(set(labels))
        for klasse in self.classes:
            # alle Sequenzen dieser Klasse rausfiltern
            klasse_seqs = [sequences[i] for i in range(len(labels)) if labels[i] == klasse]
            # zu einem langen Array zusammenfügen, lengths merken
            X = np.vstack(klasse_seqs)
            lengths = [len(seq) for seq in klasse_seqs]

            model = GaussianHMM(n_components=self.n_states, n_iter=self.n_iter, covariance_type="diag")
            model.fit(X, lengths)
            self.models[klasse] = model
        return self

    def decision_function(self, sequence):
        """Berechnet die Log-Likelihood der Sequenz für jede Klasse."""
        scores = {}
        for klasse, model in self.models.items():
            scores[klasse] = model.score(sequence)
        return scores

    def predict(self, sequence):
        """Gibt die Klasse mit dem höchsten Score zurück."""
        scores = self.decision_function(sequence)
        return max(scores, key=scores.get)

    def save(self, path: str):
        """Speichert das trainierte Modell als pickle-Datei."""
        data = {
            "models": self.models,
            "classes": self.classes,
            "n_states": self.n_states,
            "n_iter": self.n_iter,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell aus einer pickle-Datei."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        classifier = cls(n_states=data["n_states"], n_iter=data["n_iter"])
        classifier.models = data["models"]
        classifier.classes = data["classes"]
        return classifier