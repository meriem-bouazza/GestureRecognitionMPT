import numpy as np
from SignalHub import GALY, bgr, Module, get_nested_key
from collections import deque
from GestureRecognition.hotkeys import reset_requested

class TrailMarker(Module):
    """
    Modul zum Zeichnen einer Spur anhand der Bewegung eines Fingers.

    Die Position eines bestimmten Finger-Landmarks wird über mehrere Frames
    hinweg gespeichert. Aus diesen Punkten kann anschließend eine Linie
    erzeugt werden, die den Bewegungsverlauf des Fingers visualisiert.

    Ziel ist es, die Verarbeitung der Landmark-Daten sowie die Verwaltung
    eines Zustands über mehrere Frames hinweg selbst zu implementieren.
    """

    def __init__(self, outputSignal="trailmarker"):
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
        Da dieses Modul keine eigenen Daten erzeugt, reicht beispielsweise:

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
            name="trailmarker",
        )

    def start(self, data):
        """
        Initialisierung des Modulzustands.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.

        Ziel ist es, alle Variablen vorzubereiten, die während der
        Laufzeit des Moduls benötigt werden.

        Hinweise
        --------
        - Lese benötigte Parameter aus der Konfiguration.
        - Bestimme beispielsweise, welcher Finger verfolgt werden soll.
        - Lege eine Datenstruktur an, in der mehrere vergangene
          Fingerpositionen gespeichert werden können,
          z.B. :class:`collections.deque` mit einer maximalen Größe.
        - Diese Historie wird später verwendet, um eine Spur zu zeichnen.
        - Speichere aus der Konfiguration weitere benötigte Parameter,
          z.B. Finger-Index, maximale Anzahl verlorener Frames oder
          Webcam-Parameter.
        - Für den Zugriff auf verschachtelte Konfigurationswerte kann
          :meth:`get_nested_key` verwendet werden.

        .. tip::
           Eine ``deque`` ist ideal für Trajektorien,
           da sie effizient alte Punkte entfernt.

        .. note::
           Initialisiere hier nur Zustände und Parameter,
           keine eigentliche Verarbeitung.

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
        self.max_lost = get_nested_key("preprocessor.max_lost", config)
        # Wie viele Punkte der Spur sichtbar bleiben (mehr = Zeichnung bleibt länger).
        trail_length = get_nested_key("trailmarker.trail_length", config) or 400
        self.trail = deque(maxlen=trail_length)
        self.lost_count = 0
        self.color = bgr("#FFFF00")

        W = get_nested_key("webcam.width", config) or 640
        H = get_nested_key("webcam.height", config) or 360
        self.mapping = np.array([[W, 0, 0], [0, H, 0]], dtype=np.float64)
        return {}

    def step(self, data):
        if reset_requested():
            self.trail.clear()

        result = data["detector"]
        galy = GALY()
        galy.layer("trailmarker")
        galy.set_layer_affine_mapping(self.mapping)
        galy.putText("ENTF: neue Geste", (30, 700), color=(0, 255, 255))

        if result is None or not result.hand_landmarks:
            self.lost_count += 1
            if self.lost_count > self.max_lost:
                self.trail.clear()
            return {"galy": galy}

        self.lost_count = 0
        tip = result.hand_landmarks[0][self.finger_idx]
        self.trail.append((tip.x, tip.y))

        for i in range(1, len(self.trail)):
            galy.line(self.trail[i - 1], self.trail[i], self.color, 3)

        return {"galy": galy}

    def stop(self, data):
        """
        Wird aufgerufen, wenn das Modul beendet wird.

        Ziel ist es, bei Bedarf Ressourcen freizugeben oder interne
        Zustände zurückzusetzen.

        Hinweise
        --------
        - In vielen Fällen ist keine spezielle Bereinigung notwendig.

        .. note::
           Diese Methode ist optional, kann aber sinnvoll sein,
           wenn Zustände explizit zurückgesetzt werden sollen.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        pass