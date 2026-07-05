import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset():

    sequences = [np.random.randn(30,2) for _ in range(10)]
    labels = ["A"]* 5 + ["B"] * 5 
    classes = sorted(set(labels))
    fig, axes = plt.subplots(1, len(classes), figsize=(5* len(classes), 4))
    for i, cls in enumerate (classes):
        ax = axes[i]
        for seq, lbl in zip(sequences, labels):
            if lbl == cls:
                ax.plot(seq[:,0], seq[:,1])
        ax.set_title(f"Klasse {cls}")
    
    plt.show()
    """
    TODO: Visualisierung des eigenen Datensatzes

    Ziel:
    -----
    Entwickle eine Möglichkeit, deinen aufgenommenen Datensatz visuell zu
    inspizieren und zu verstehen.

    Warum ist das wichtig?
    ----------------------
    - Du musst nachvollziehen können, was dein Modell eigentlich „sieht“
    - Fehler im Datensatz lassen sich visuell oft sofort erkennen
    - Qualität der Daten ist entscheidend für die Modellperformance

    Anforderungen / Ideen:
    ----------------------
    - Lade deinen Trainingsdatensatz
    - Visualisiere mehrere Sequenzen pro Klasse
    - Stelle sicher, dass:
        - unterschiedliche Gesten klar unterscheidbar sind
        - Sequenzen sinnvoll aussehen (keine Ausreißer, keine leeren Daten)

    .. tip::
       Ein einfacher Ansatz:
         - Plotte Trajektorien (z. B. x/y-Koordinaten)
         - Zeige mehrere Beispiele pro Klasse übereinander

    .. note::
       Du kannst selbst entscheiden:
         - Wie viele Sequenzen du anzeigst
         - Welche Features du visualisierst
         - Ob du interaktive Elemente einbaust

    .. tip::
       Interaktivität (z. B. Klick auf eine Sequenz) kann hilfreich sein,
       um einzelne Beispiele genauer zu untersuchen.

    Abgabe:
    -------
    - Du musst in der Lage sein, deinen Datensatz visuell zu präsentieren
    - Du solltest erklären können:
        - Wie unterscheiden sich die Klassen?
        - Gibt es problematische Beispiele?

    Erweiterung (optional):
    -----------------------
    - Mittelwerte oder typische Sequenzen pro Klasse darstellen
    - Ausreißer automatisch erkennen
    """


def evaluate_classifier():
    """Trainiert HMMClassifier, berechnet Accuracy und Confusion Matrix."""
    from GestureRecognition.hmmclassifier import HMMClassifier
    from GestureRecognition.labeling import dataset_building

    # alle Aufnahmen aus data/recordings/ laden
    sequences, labels = dataset_building("data/recordings")

    # 80% Training, 20% Test pro Klasse
    train_seqs, train_labels, test_seqs, test_labels = \
        HMMClassifier.train_test_split(sequences, labels)
    print(f"Train: {len(train_seqs)} Sequenzen")
    print(f"Test:  {len(test_seqs)} Sequenzen")

    # Modell trainieren und als data/hmm.pkl speichern
    clf = HMMClassifier(n_states=7, n_iter=100)
    clf.fit(train_seqs, train_labels)
    clf.save("data/hmm.pkl")
    print("Modell gespeichert: data/hmm.pkl")

    # Vorhersagen auf Testdaten
    y_pred = [clf.predict(s) for s in test_seqs]
    correct = sum(t == p for t, p in zip(test_labels, y_pred))
    print(f"Accuracy: {correct}/{len(test_labels)} = "
          f"{correct/len(test_labels):.1%}")

    # Confusion Matrix berechnen
    classes = sorted(set(labels))
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(test_labels, y_pred):
        if pred in idx:
            cm[idx[true]][idx[pred]] += 1

    # Confusion Matrix plotten
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Vorhersage")
    ax.set_ylabel("Tatsaechlich")
    ax.set_title(
        f"Confusion Matrix - Accuracy: {correct/len(test_labels):.1%}"
    )

    # Werte in Matrix eintragen
    threshold = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion Matrix gespeichert: confusion_matrix.png")
    plt.show()



def replay_recordings(recordings_dir="data/recordings", label=None,
                      max_examples=3, interval=0.02):
    """Spielt aufgenommene Trajektorien Punkt für Punkt ab (Qualitaetskontrolle).

    Laedt die .npy-Aufnahmen und zeichnet jede Geste animiert nach: eine
    wachsende Linie plus roter "Stift"-Punkt. So lassen sich fehlerhafte oder
    inkonsistente Aufnahmen schnell erkennen.

    Parameters
    ----------
    recordings_dir : str
        Ordner mit den Aufnahmen (Unterordner = Label, darin .npy-Dateien).
    label : str, optional
        Nur diese Geste abspielen (z.B. "A"). None = alle Klassen.
    max_examples : int
        Maximale Anzahl Beispiele pro Klasse (0 = alle).
    interval : float
        Pause pro Frame in Sekunden (kleiner = schnelleres Replay).
    """
    from pathlib import Path

    recordings_dir = Path(recordings_dir)

    # Aufnahmen sammeln, pro Klasse auf max_examples begrenzen
    playlist, counts = [], {}
    for npy_file in sorted(recordings_dir.glob("*/*.npy")):
        lbl = npy_file.parent.name
        if label is not None and lbl != label:
            continue
        counts[lbl] = counts.get(lbl, 0) + 1
        if not max_examples or counts[lbl] <= max_examples:
            playlist.append((lbl, npy_file))

    if not playlist:
        print("Keine Aufnahmen gefunden"
              + (f" fuer Label '{label}'." if label else "."))
        return

    n_classes = len({lbl for lbl, _ in playlist})
    print(f"{len(playlist)} Aufnahmen ({n_classes} Klassen) — "
          "Fenster schliessen zum Abbrechen.")

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    line, = ax.plot([], [], "-", color="royalblue", lw=2)
    dot, = ax.plot([], [], "o", color="red", ms=8)

    for lbl, npy_file in playlist:
        if not plt.fignum_exists(fig.number):
            break  # Fenster wurde geschlossen

        traj = np.load(npy_file)
        if traj.ndim != 2 or traj.shape[0] < 2 or traj.shape[1] < 2:
            print(f"Uebersprungen (ungueltige Form {traj.shape}): {npy_file.name}")
            continue
        x, y = traj[:, 0], traj[:, 1]

        # Achsen an diese Aufnahme anpassen; y invertieren (Bildkoordinaten)
        ax.set_title(f"Klasse '{lbl}' — {npy_file.name}  ({len(x)} Punkte)")
        m = 0.05
        ax.set_xlim(x.min() - m, x.max() + m)
        ax.set_ylim(y.max() + m, y.min() - m)
        ax.set_aspect("equal")

        for i in range(len(x)):
            if not plt.fignum_exists(fig.number):
                break
            line.set_data(x[:i + 1], y[:i + 1])
            dot.set_data([x[i]], [y[i]])
            plt.pause(interval)
        plt.pause(0.4)  # fertige Geste kurz stehen lassen

    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    print("Replay beendet.")