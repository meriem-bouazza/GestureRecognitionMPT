import numpy as np
from scipy.stats import multivariate_normal
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

    def forward(self, sequence, A, pi, means, covs) -> float:
        T = len(sequence)  # Sequenzlänge
        N = len(pi)        # Anzahl Zustände

        # Emissionswahrscheinlichkeit p(yt|xt) für jeden Zeitschritt und Zustand
        B = np.array([
            [multivariate_normal.pdf(sequence[t], means[j], covs[j]) for j in range(N)]
            for t in range(T)
        ])  

        alpha = np.zeros((T, N))
        alpha[0] = pi * B[0]  #Startverteilung * erste Beobachtung

        for t in range(1, T):
            # Summe über alle Vorgängerzustände
            alpha[t] = (alpha[t-1] @ A) * B[t]

        # Summe über alle Endzustände → log P(O|λ)
        return np.log(alpha[-1].sum())
    
    def baum_welch(self, sequences):
        """Lernt A, pi, means, covs aus Trainingssequenzen einer Klasse."""
        N = self.n_states
        n_features = sequences[0].shape[1]  # = 2 (x, y)

        # Parameter zufällig initialisieren
        A = np.ones((N, N)) / N
        pi = np.ones(N) / N
        means = np.random.randn(N, n_features)
        covs = np.array([np.eye(n_features)] * N)

        for _ in range(self.n_iter):

            # Statistiken zurücksetzen
            A_num = np.zeros((N, N))
            pi_num = np.zeros(N)
            means_num = np.zeros((N, n_features))
            covs_num = np.zeros((N, n_features, n_features))
            gamma_sum = np.zeros(N)

            for seq in sequences:
                T = len(seq)

                # Emissionswahrscheinlichkeiten p(yt|xt)
                B = np.array([
                    [multivariate_normal.pdf(seq[t], means[j], covs[j]) for j in range(N)]
                    for t in range(T)
                ])

                # Forward
                alpha = np.zeros((T, N))
                alpha[0] = pi * B[0]
                for t in range(1, T):
                    alpha[t] = (alpha[t-1] @ A) * B[t]

                # Backward 
                beta = np.ones((T, N))
                for t in range(T-2, -1, -1):
                    beta[t] = A @ (B[t+1] * beta[t+1])

                # Gamma: wie wahrscheinlich ist Zustand j zum Zeitpunkt t?
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)

                # Xi: wie wahrscheinlich ist Übergang i→j zum Zeitpunkt t?
                xi = np.zeros((T-1, N, N))
                for t in range(T-1):
                    xi[t] = alpha[t][:, None] * A * B[t+1] * beta[t+1]
                    xi[t] /= xi[t].sum()

                # Statistiken aufsammeln
                pi_num += gamma[0]
                A_num += xi.sum(axis=0)
                gamma_sum += gamma.sum(axis=0)
                for j in range(N):
                    means_num[j] += (gamma[:, j:j+1] * seq).sum(axis=0)
                    diff = seq - means[j]
                    covs_num[j] += (gamma[:, j, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)

            # M-Step: Parameter neu berechnen
            pi = pi_num / pi_num.sum()
            A = A_num / A_num.sum(axis=1, keepdims=True)
            means = means_num / gamma_sum[:, None]
            covs = covs_num / gamma_sum[:, None, None]

        return A, pi, means, covs

    def fit(self, sequences: list, labels: list):
        """
        TODO: Trainiere den Klassifikator

        Ziel:
        -----
        Trainiere ein separates HMM für jede Klasse basierend auf den
        gegebenen Sequenzen.


        Anforderungen / Ideen:
        ----------------------
        - Zerlege die Daten so, dass du pro Klasse alle Sequenzen bekommst
        - Trainiere ein Modell pro Klasse
        - Speichere die trainierten Modelle intern

        .. tip::
           Überlege dir eine sinnvolle Datenstruktur wie:
           ``label -> (Daten, Sequenzlängen)``

        .. note::
           Die konkrete Umsetzung ist offen:
            - Wie genau du Daten aufteilst
            - Wie du dein Modell initialisierst
            - Welche Hyperparameter du verwendest

        .. warning::
           Achte darauf, dass:
            - ``lengths`` zu ``X`` passen
            - Labels korrekt zu Sequenzen zugeordnet sind

        Erweiterung:
        ------------
        - Experimentiere mit verschiedenen Modellgrößen
        - Nutze eine Grid Search zur Optimierung
        - Verwende ein separates Testset zur Evaluation

        Returns
        -------
        self
        """
        sequences = [np.array(seq) for seq in sequences]  # sicherstellen dass alles numpy ist
        self.classes = list(set(labels))
        for klasse in self.classes:
            # alle Sequenzen dieser Klasse rausfiltern
            klasse_seqs = [sequences[i] for i in range(len(labels)) if labels[i] == klasse]
            # HMM für diese Klasse trainieren
            A, pi, means, covs = self.baum_welch(klasse_seqs)
            # Modell speichern
            self.models[klasse] = {"A": A, "pi": pi, "means": means, "covs": covs}
        return self

    def decision_function(self, sequence):
        """
        TODO: Berechne Scores für jede Klasse

        Ziel:
        -----
        Berechne für jede Eingabesequenz einen Score pro Klasse
        (z. B. Log-Likelihood unter jedem Modell).

        Anforderungen / Ideen:
        ----------------------
        - Zerlege die Eingabe in einzelne Sequenzen
        - Berechne für jede Sequenz:
            Score unter jedem Klassenmodell
        - Gib eine Struktur zurück wie:
            ``(n_sequences, n_classes)``

        .. tip::
           Die meisten HMM-Implementierungen bieten eine
           ``score``-Funktion für Likelihoods.

        .. note::
           Du entscheidest selbst:
            - Welcher Score verwendet wird
            - Wie du mehrere Sequenzen behandelst

        .. warning::
           Stelle sicher, dass:
            - Die Reihenfolge der Klassen konsistent ist
            - Scores vergleichbar sind

        Returns
        -------
        scores : array-like
            Score pro Sequenz und Klasse
        """
        # Log-Likelihood für jede Klasse berechnen
        scores = {}
        for klasse, model in self.models.items():
            scores[klasse] = self.forward(sequence, model["A"], model["pi"], model["means"], model["covs"])
        return scores

    def predict(self, sequence):
        """
        TODO: Sage Klassenlabels voraus

        Ziel:
        -----
        Weise jeder Eingabesequenz ein Label zu.

        Anforderungen / Ideen:
        ----------------------
        - Nutze deine ``decision_function``
        - Wähle für jede Sequenz die Klasse mit bestem Score

        .. tip::
           Typischerweise:
           ``argmax über Klassen``

        .. note::
           Achte darauf, dass:
            - Klassenreihenfolge konsistent ist
            - Rückgabewerte klar interpretierbar sind

        Erweiterung:
        ------------
        - Gib zusätzlich Unsicherheiten oder Scores zurück
        - Implementiere Top-k Vorhersagen

        Returns
        -------
        labels : list
            Vorhergesagte Labels
        """
        scores = self.decision_function(sequence)
        # Klasse mit höchstem Score zurückgeben
        return max(scores, key=scores.get)

    def save(self, path: str):
        import pickle
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
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        classifier = cls(n_states=data["n_states"], n_iter=data["n_iter"])
        classifier.models = data["models"]
        classifier.classes = data["classes"]
        return classifier