import numpy as np
from SignalHub import get_nested_key, Module
from collections import deque
from GestureRecognition.labeling import (
    resample_by_arc_length,
    add_velocity_features,
    add_curvature_feature,
)
from GestureRecognition.hotkeys import reset_requested

class Preprocessor(Module):
    """
    Modul zur Vorverarbeitung von Fingertrajektorien.

    Dieses Modul verarbeitet die vom Handdetektor gelieferten Landmarken
    und extrahiert daraus die Bewegung eines bestimmten Fingers über
    mehrere Frames hinweg.

    Ziel ist es, eine Trajektorie zu sammeln, diese zu normalisieren
    und anschließend als Eingabe für nachfolgende Module bereitzustellen.
    """

    def __init__(self, outputSignal="preprocessor"):
        """
        Konstruktor des Moduls.

        Ziel ist es, das Modul beim Framework korrekt zu registrieren.

        Hinweise
        --------
        - Ein Modul muss definieren, **welche Signale es empfangen möchte**.
        - Diese werden über ``inputSignals`` angegeben.
        - Nur Signale, die hier subscribed werden, erscheinen später im
          ``data`` Dictionary der Methoden :meth:`start` und :meth:`step`.

        Für dieses Modul werden unter anderem folgende Signale benötigt:

        - ``config`` : Systemkonfiguration
        - ``detector`` : Ergebnisse der Handdetektion

        Zusätzlich muss ein **Output-Schema** definiert werden.

        Output Schema
        -------------
        Das Modul erzeugt ein Signal mit dem Namen ``preprocessor``.

        Dieses Signal enthält entweder eine normalisierte Trajektorie
        oder ``None``, falls noch nicht genügend Daten gesammelt wurden.

        Beispiel:

        ``outputSchema={"type": "object", "properties": {outputSignal: {}}}``

        .. note::
           Die Basisklasse :class:`Module` erwartet beim Aufruf von
           ``super().__init__`` unter anderem:

           - ``inputSignals``
           - ``outputSchema``
           - ``name`` des Moduls

        Parameters
        ----------
        outputSignal : str, optional
            Name des erzeugten Output-Signals.
        """
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="preprocessor",
        )

    def start(self, data):
        """
        Initialisierung des Modulzustands.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.

        Ziel ist es, alle benötigten Parameter aus der Konfiguration zu
        lesen und interne Datenstrukturen vorzubereiten.

        Hinweise
        --------
        - Lese relevante Parameter aus der Konfiguration, z.B.
          den zu verfolgenden Finger.
        - Lege eine Datenstruktur an, um mehrere vergangene
          Fingerpositionen zu speichern, z.B. :class:`collections.deque`
          mit einer maximalen Größe.
        - Speichere außerdem Parameter wie die maximale Anzahl
          verlorener Frames oder die minimale Anzahl benötigter Punkte.
        - Zum Zugriff auf verschachtelte Konfigurationswerte kann
          :meth:`get_nested_key` verwendet werden.

        .. tip::
            Eine ``deque`` mit fester Länge ist ideal für Trajektorien,
            da alte Punkte automatisch verworfen werden.

        .. note::
            Trenne klar zwischen:
              - Initialisierung von Parametern (``start``)
              - Verarbeitung von Daten (``step``)

        Parameters
        ----------
        data : dict
            Eingabedaten des Frameworks. Enthält unter anderem das
            Signal ``config``.

        Returns
        -------
        dict
            Ein leeres Dictionary.
        """
        config = data["config"]
        self.finger_idx = get_nested_key("preprocessor.finger_idx", config)
        self.buffer_size = get_nested_key("preprocessor.buffer_size", config)
        self.max_lost = get_nested_key("preprocessor.max_lost", config)
        self.min_steps = get_nested_key("preprocessor.min_steps", config)
        self.trajectory = deque(maxlen=self.buffer_size)
        self.lost_count = 0
        return {}

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Ziel ist es, eine Fingerposition aus den erkannten Landmarken
        zu extrahieren und diese in einer Trajektorie zu speichern.

        Hinweise
        --------
        - Greife auf das ``detector`` Signal zu, um erkannte
          Handlandmarks zu erhalten.
        - Falls keine Hand erkannt wurde, sollte ein interner
          Zähler für verlorene Frames erhöht werden.
        - Wird eine Hand erkannt, kann die Landmarke des gewünschten
          Fingers extrahiert werden.
        - Die Position dieses Fingers kann anschließend in einer
          Trajektorie gespeichert werden.
        - Sobald genügend Punkte gesammelt wurden, kann die
          Trajektorie weiterverarbeitet werden.

        Mögliche Verarbeitungsschritte:

        - Umwandlung der gespeicherten Punkte in ein
          :class:`numpy.ndarray`
        - Berechnung eines Zentrums der Trajektorie
        - Skalierung oder Normalisierung der Punkte

        .. tip::
            Arbeite schrittweise:
              1. Prüfen, ob Landmarken vorhanden sind
              2. Fingerposition extrahieren
              3. In Trajektorie speichern
              4. Optional normalisieren

        .. warning::
            Achte darauf, dass:
              - genügend Punkte vorhanden sind
              - keine fehlerhaften Frames verarbeitet werden
              - verlorene Frames sinnvoll behandelt werden

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``detector`` : erkannte Hände und Landmarken
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            Gibt entweder ``None`` oder eine normalisierte Trajektorie
            zurück.

            Beispiel:

            ``return {outputSignal: trajectory}``
        """
        if reset_requested():
            self.trajectory.clear()
            self.lost_count = 0
            return {"preprocessor": None}

        result = data["detector"]

        if result is None or not result.hand_landmarks:
            self.lost_count += 1
            if self.lost_count > self.max_lost:
                self.trajectory.clear()
            return {"preprocessor": None}

        self.lost_count = 0
        tip = result.hand_landmarks[0][self.finger_idx]
        self.trajectory.append((tip.x, tip.y))

        if len(self.trajectory) < self.min_steps:
            return {"preprocessor": None}

        traj = np.array(self.trajectory, dtype=np.float32)
        center = traj.mean(axis=0)
        traj -= center
        scale = max(traj[:, 0].max() - traj[:, 0].min(),
                    traj[:, 1].max() - traj[:, 1].min())
        if scale > 0:
            traj /= scale   

        traj = resample_by_arc_length(traj, n_points=50)
        traj = add_velocity_features(traj)
        traj = add_curvature_feature(traj)
        return {"preprocessor": traj}

    def stop(self, data):
        """
        Wird aufgerufen, wenn das Modul beendet wird.

        Ziel ist es, bei Bedarf interne Zustände zurückzusetzen
        oder Ressourcen freizugeben.

        Hinweise
        --------
        - In vielen Fällen ist keine spezielle Bereinigung notwendig.

        .. note::
           Diese Methode ist optional, kann aber relevant werden,
           wenn interne Zustände explizit zurückgesetzt werden sollen.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        pass