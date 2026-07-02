import math

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from SignalHub import GALY, bgr, get_nested_key, Module

mp_hand = mp.tasks.vision.HandLandmarksConnections


class _OneEuroFilter:
    """Adaptives Glätten eines skalaren Signals (One-Euro-Filter).

    Wenig Verzögerung bei schnellen Bewegungen, starke Glättung bei Ruhe –
    ideal gegen das Zittern der MediaPipe-Landmarks.
    """

    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


def draw_hand_landmarks(hand_landmarks, galy: GALY):
    lm = {
        "thumb":         {"color": bgr("#0000FF")},
        "index_finger":  {"color": bgr("#00FF00")},
        "middle_finger": {"color": bgr("#FF0000")},
        "ring_finger":   {"color": bgr("#00FFFF")},
        "pinky_finger":  {"color": bgr("#FF00FF")},
        "palm":          {"color": bgr("#C8C8C8")},
    }
    x = np.inf
    y = np.inf
    for key in lm.keys():
        pts = set()
        for conn in getattr(mp_hand, f"HAND_{key.upper()}_CONNECTIONS"):
            start = (hand_landmarks[conn.start].x,
                    hand_landmarks[conn.start].y)
            end = (hand_landmarks[conn.end].x,
                hand_landmarks[conn.end].y)
            x = min(x, start[0], end[0])
            y = min(y, start[1], end[1])
            galy.line(start, end, lm[key]["color"], 2)
            pts.update([conn.start, conn.end])
        for pt in pts:
            galy.circle((hand_landmarks[pt].x, hand_landmarks[pt].y), 5, (255,255,255), 1)
            galy.circle((hand_landmarks[pt].x, hand_landmarks[pt].y), 4, lm[key]["color"], -1)


class HandDetector(Module):
    """
    Modul zur Erkennung von Händen und deren Landmarken.

    Dieses Modul verwendet das MediaPipe Hand Landmarker Modell, um Hände
    in einem Kamerabild zu erkennen und deren Landmarken zu bestimmen.

    Ziel ist es, die Webcam-Bilder zu verarbeiten, eine Handdetektion
    durchzuführen und die erkannten Landmarken sowie eine Visualisierung
    an das Framework zurückzugeben.
    """

    def __init__(self, outputSignal="detector"):
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
        - ``webcam`` : aktuelles Kamerabild

        Zusätzlich muss ein **Output-Schema** definiert werden.

        Output Schema
        -------------
        Das Modul erzeugt ein Signal mit dem Namen ``detector``.

        Dieses Signal enthält das Ergebnis der Handdetektion, welches
        beispielsweise Informationen über erkannte Hände und Landmarken
        enthalten kann.

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
            inputSignals=["config", "webcam"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="detector",
        )

    def start(self, data):
        """
        Initialisierung des Moduls.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.

        Ziel ist es, das benötigte Handdetektionsmodell zu laden und
        für die spätere Verarbeitung vorzubereiten.

        Hinweise
        --------
        - MediaPipe stellt eine Hand-Landmark-Erkennung
          `bereit <https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb>`_.
        - Laden sie wie im Artikel beschrieben das Modell ein und speichern sie das detector
          Objekt in einem Attribut des Moduls. z.B. ``self.detector``

        .. tip::
           Halte die Initialisierung strikt getrennt von der Verarbeitung.
           In ``start`` sollte nur vorbereitet, nicht gerechnet werden.

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
        import urllib.request, os
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                model_path,
            )
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=get_nested_key(
                "config.detector.min_detection_confidence", data) or 0.6,
            min_hand_presence_confidence=get_nested_key(
                "config.detector.min_presence_confidence", data) or 0.6,
            min_tracking_confidence=get_nested_key(
                "config.detector.min_tracking_confidence", data) or 0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # VIDEO-Modus braucht streng steigende Zeitstempel (in ms).
        self.frame_interval_ms = 33  # ~30 FPS
        self.timestamp_ms = 0

        # One-Euro-Glättung gegen Landmark-Jitter: ein Filter je x/y pro Landmark.
        freq = get_nested_key("config.detector.smoothing.freq", data) or 30.0
        min_cutoff = get_nested_key("config.detector.smoothing.min_cutoff", data) or 1.0
        beta = get_nested_key("config.detector.smoothing.beta", data) or 0.5
        self.smoothers = [
            (_OneEuroFilter(freq, min_cutoff, beta),
             _OneEuroFilter(freq, min_cutoff, beta))
            for _ in range(21)
        ]

        W = get_nested_key("config.webcam.width", data) or 640
        H = get_nested_key("config.webcam.height", data) or 360
        self.mapping = np.array([[W, 0, 0], [0, H, 0]], dtype=np.float64)
        return {}

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Ziel ist es, ein Kamerabild zu analysieren, Hände zu erkennen und
        deren Landmarken zu bestimmen.

        Hinweise
        --------
        - Greife auf das ``webcam`` Signal zu, um das aktuelle Bild zu erhalten.
        - Das Bild liegt typischerweise als :class:`np.ndarray` vor.
        - Für MediaPipe muss das Bild ggf. in ein geeignetes Format
          konvertiert werden (:class:`mp.Image`).
        - Anschließend kann das Bild an den Handdetektor übergeben werden.
        - Das Ergebnis enthält Informationen über erkannte Hände sowie
          deren Landmarken.
        - Für jede erkannte Hand können die Landmarken anschließend
          visualisiert werden.
        - Für die Visualisierung kann ein :class:`GALY` Objekt verwendet werden.
        - Die Funktion :func:`draw_hand_landmarks` kann genutzt werden,
          um Landmarken und Verbindungen darzustellen.

        .. tip::
           Arbeite schrittweise:
            1. Bild holen
            2. Format konvertieren
            3. Detektion durchführen
            4. Ergebnis verarbeiten / visualisieren

        .. warning::
            Achte darauf, dass:
                - das Bildformat korrekt ist (RGB vs. BGR)
                - die Detektion pro Frame effizient bleibt (Live-Demo)

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``webcam`` : aktuelles Kamerabild
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            Soll das Ergebnis der Handdetektion sowie optional ein
            :class:`GALY` Objekt für die Visualisierung enthalten.

            Beispiel:

            ``return {outputSignal: result, "galy": galy}``
        """
        frame = data["webcam"]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO-Modus: Landmarks werden über Frames getrackt (stabiler, schneller).
        self.timestamp_ms += self.frame_interval_ms
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        if result.hand_landmarks:
            # Jitter der x/y-Landmarks glätten (nutzen z.B. Preprocessor/TrailMarker).
            for lm, (fx, fy) in zip(result.hand_landmarks[0], self.smoothers):
                lm.x = fx(lm.x)
                lm.y = fy(lm.y)
        else:
            # Hand verloren -> Filterzustand zurücksetzen, damit die Spur nicht springt.
            for fx, fy in self.smoothers:
                fx.reset()
                fy.reset()

        galy = GALY()
        galy.layer("detector")
        galy.set_layer_affine_mapping(self.mapping)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                draw_hand_landmarks(hand, galy)

        return {"detector": result, "galy": galy}

    def stop(self, data):
        """
        Wird aufgerufen, wenn das Modul beendet wird.

        Ziel ist es, bei Bedarf Ressourcen freizugeben oder interne
        Zustände zurückzusetzen.

        Hinweise
        --------
        - In vielen Fällen ist keine spezielle Bereinigung notwendig.

        .. note::
           Diese Methode ist optional, kann aber wichtig werden,
           wenn externe Ressourcen (z. B. Modelle, Streams) verwendet werden.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()