import pickle
import warnings
import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMClassifier:
    """
    HMM-basierter Klassifikator für Gestentrajektorien.

    Pro Klasse wird ein GaussianHMM trainiert. Klassifikation läuft über
    den argmax der Log-Likelihoods aller Klassenmodelle (Forward-Algorithmus).
    """

    def __init__(self, n_states: int = 5, n_iter: int = 100, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state  # feste Seed für Reproduzierbarkeit
        self.models = {}
        self.classes = []

    def fit(self, sequences: list, labels: list):
        """Trainiert ein GaussianHMM pro Klasse mit hmmlearn."""
        sequences = [np.array(seq) for seq in sequences]
        self.classes = list(set(labels))

        for klasse in self.classes:
            klasse_seqs = [sequences[i] for i in range(len(labels)) if labels[i] == klasse]

            # hmmlearn braucht alle Sequenzen gestapelt + lengths damit es
            # keine falschen Übergänge zwischen Aufnahmen lernt
            X = np.vstack(klasse_seqs)
            lengths = [len(seq) for seq in klasse_seqs]

            # n_states darf nie größer als die kürzeste Sequenz sein
            n = min(self.n_states, min(lengths))

            model = GaussianHMM(
                n_components=n,
                n_iter=self.n_iter,
                covariance_type="diag",  # robuster bei wenig Daten als "full"
                random_state=self.random_state,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # ConvergenceWarning unterdrücken
                    model.fit(X, lengths)
                self.models[klasse] = model
            except Exception:
                # singuläre Kovarianz bei zu wenig Daten -> None, Score -inf
                print(f"Warnung: Modell für '{klasse}' konnte nicht trainiert werden.")
                self.models[klasse] = None

        return self

    def decision_function(self, sequence):
        """Berechnet Log-Likelihood der Sequenz für jede Klasse."""
        scores = {}
        for klasse, model in self.models.items():
            if model is None:
                scores[klasse] = -np.inf
                continue
            try:
                scores[klasse] = model.score(sequence)
            except Exception:
                scores[klasse] = -np.inf
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
            "random_state": self.random_state,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        """Lädt ein gespeichertes Modell aus einer pickle-Datei."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        classifier = cls(
            n_states=data["n_states"],
            n_iter=data["n_iter"],
            random_state=data.get("random_state", 42),
        )
        classifier.models = data["models"]
        classifier.classes = data["classes"]
        return classifier

    @staticmethod
    def train_test_split(sequences, labels, test_ratio=0.2, seed=42):
        """Teilt Daten pro Klasse in Trainings- und Testsequenzen."""
        rng = np.random.default_rng(seed)
        train_seqs, train_labels = [], []
        test_seqs, test_labels = [], []

        classes = list(set(labels))
        for klasse in classes:
            idx = [i for i in range(len(labels)) if labels[i] == klasse]
            idx = rng.permutation(idx)  # mischen für fairen Split

            n_test = max(1, int(len(idx) * test_ratio))
            for i in idx[n_test:]:
                train_seqs.append(sequences[i])
                train_labels.append(labels[i])
            for i in idx[:n_test]:
                test_seqs.append(sequences[i])
                test_labels.append(labels[i])

        return train_seqs, train_labels, test_seqs, test_labels