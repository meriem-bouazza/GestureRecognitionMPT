# Projektdokumentation: Gesture Recognition

**GestureRecognitionMPT** 

In diesem Projekt haben wir zu fünft ein System gebaut, das mit dem
Zeigefinger in die Luft gezeichnete Buchstaben (A–Z) erkennt. Die
Handbewegung wird per Kamera getrackt, daraus werden Features berechnet
und ein Hidden-Markov-Modell pro Klasse übernimmt die Klassifikation.

Ein Ergebnis hat uns dabei überrascht: Ein zusätzliches Geschwindigkeits-
Feature hat das Modell zunächst schlechter gemacht (72,6 % statt 78,5 %
Genauigkeit). Erst als wir es mit einem Krümmungs-Feature kombiniert haben,
wurde das Modell besser und vor allem deutlich zuverlässiger (die Streuung
zwischen verschiedenen Test-Splits sank von ±4,6 % auf ±1,3 %).

---

## 1. Wer hat was gebaut

| Person | Verantwortung | Kernbeitrag |
|---|---|---|
| Melina | HandDetector | MediaPipe-Landmark-Extraktion, Behandlung fehlender Hand-Erkennung |
| Kardelen Atin | Preprocessor, TrailMarker | Trajektorien-Puffer, Live-Visualisierung, Abgleich mit Trainings-Features |
| Sabrina Akouz | Labeling & Datensatz | Aufnahme-Tooling, dataset_building(), Geschwindigkeits- und Krümmungs-Features |
| Meriem Bouazza | HMM-Klassifikator | GaussianHMM-Wrapper, Reproduzierbarkeit, Rückwärts-Training |
| Vipusiny Vijayakumar | HMM-Runtime & Auswertung | Live-Inferenz-Modul, Visualisierung, Confusion Matrix |

Weil Training und Live-Erkennung in getrennten Dateien von unterschiedlichen
Personen gepflegt werden, mussten wir uns im Team explizit darauf einigen,
dass beide Seiten die Daten genau gleich verarbeiten. Das kam zum Beispiel
zum Tragen, als die Krümmungs-Features in den Live-Preprocessor
nachgezogen werden mussten, nachdem sie im Training schon eingebaut waren.

## 2. Systemüberblick
Fünf Module geben ein Signal über das Framework SignalHub an das nächste
weiter:

1. Webcam liefert das Kamerabild.
2. HandDetector (MediaPipe) extrahiert die Handlandmarken.
3. TrailMarker zeichnet die Fingerspur zur Kontrolle ins Bild.
4. Preprocessor sammelt und normalisiert die Spur des Zeigefingers und
   berechnet daraus den Feature-Vektor.
5. HMMModule bewertet den Vektor gegen die trainierten Klassenmodelle und
   gibt Label und Score zurück.

Jedes Modul bekommt dabei nur die Signale, die es braucht. Die HMM-Stufe
sieht zum Beispiel nie ein Rohbild, sondern nur die fertige Trajektorie.

## 3. Datenbasis
Der abgegebene Datensatz umfasst 26 Klassen, die Buchstaben A–Z. Insgesamt
sind das 784 Aufnahmen, im Schnitt etwa 30 pro Buchstabe (fünf Personen mit
je sechs Aufnahmen, teils mehrfach nachjustiert). Jede Aufnahme ist eine
rohe (T, 2)-Fingerspur, gespeichert unter
`data/recordings/<KLASSE>/<timestamp>.npy`.

Bei fünf Personen kommen zwangsläufig unterschiedlich saubere Aufnahmen
zusammen. Über die Git-Historie ließ sich nachvollziehen, wer welche Datei
ursprünglich committet hatte, sodass wir erkennbar fehlerhafte Aufnahmen
(falsche Geste gezeichnet, zu lange oder zappelige Spur) gezielt der
jeweiligen Person zuordnen und durch saubere Wiederholungen ersetzen
konnten, statt den ganzen Datensatz von Hand durchzugehen.

## 4. Von der Rohspur zum Feature-Vektor
Die Rohspur hat je nach Zeichentempo eine unterschiedliche Länge. Bevor sie
ins Modell geht, durchläuft sie ein paar Schritte, und zwar in Training und
Live über dieselbe Funktion, nicht über zwei getrennt gepflegte
Implementierungen:

1. Zentrieren und Skalieren auf die Bounding-Box der Geste.
2. Resampling entlang der Bogenlänge auf 50 Punkte. Das macht die
   Darstellung unabhängig davon, wie schnell oder langsam gezeichnet wurde.
3. Geschwindigkeit (Δx, Δy zum vorherigen Punkt) als zusätzliche Dimension.
4. Krümmung (Richtungsänderung zwischen zwei Bewegungssegmenten) als fünfte
   Dimension.
5. Für das Training zusätzlich: jede Sequenz wird auch rückwärts
   durchlaufen mit ins Training gegeben. Das verdoppelt den effektiven
   Datensatz auf 1568 Sequenzen und macht das Modell unabhängiger vom
   Startpunkt einer Geste.

Am Ende ist der Vektor pro Zeitschritt fünfdimensional: (x, y, Δx, Δy,
Krümmung).

## 5. Klassifikationsmodell
Wir trainieren ein GaussianHMM (aus der Bibliothek hmmlearn) pro Klasse.
Die versteckten Zustände stehen dabei grob für Phasen der Bewegung, also
Start, Mitte und Ende einer Geste.

- n_states = 8, n_iter = 100, random_state = 42 (fest gesetzt, damit
  Ergebnisse zwischen Läufen vergleichbar bleiben).
- Für jede Klasse werden alle zugehörigen Sequenzen gestapelt trainiert,
  zusammen mit einem Längenvektor, damit hmmlearn keine Übergänge
  zwischen unabhängigen Aufnahmen lernt.
- Eine neue Spur wird gegen alle 26 Modelle bewertet (Forward-Algorithmus,
  Log-Likelihood), und die Klasse mit dem höchsten Wert gewinnt.

## 6. Was wir bei der Genauigkeit herausgefunden haben

#### Hilft festes Resampling überhaupt?
Ja, deutlich. Schon die reine,
resamplete (x,y)-Position kam auf 78,5 % (±4,6 % über fünf Test-Splits),
im Vergleich zu spürbar instabileren Werten ohne einheitliche
Sequenzlänge.

#### Machen mehr Features die Erkennung automatisch besser?
Nicht
unbedingt, das war für uns die eigentliche Überraschung. Wir haben das
isoliert getestet, mit identischen Daten und nur unterschiedlicher
Vorverarbeitung (n = 784, 26 Klassen, 5 Seeds):

| Feature-Stufe | Test (Seed 42) | Mittel ± Std über 5 Splits |
|---|---|---|
| nur Position (x,y) | 73,9 % | 78,5 % ± 4,6 % |
| + Geschwindigkeit | 65,0 % | 72,6 % ± 4,5 % (schlechter) |
| + Geschwindigkeit + Krümmung | 80,9 % | 82,4 % ± 1,3 % (besser) |

Geschwindigkeit allein scheint eher das Rauschen zwischen einzelnen
Frames zu verstärken, als brauchbares Signal zu liefern. Erst zusammen
mit der Krümmung, die eher die Form der Bewegung beschreibt als ihre
Feinheiten, kippt der Effekt ins Positive.

#### Ist die Zahl überhaupt belastbar?
Ein einzelner Split hätte hier in
die Irre führen können. Interessanter als der Sprung im Mittelwert
(78,5 % auf 82,4 %) fanden wir eigentlich den Rückgang der Streuung von
±4,6 % auf ±1,3 %. Das Modell liefert damit nicht nur im Schnitt bessere,
sondern vor allem gleichmäßigere Ergebnisse.

Zum Vergleich: das komplette System mit Rückwärts-Training (1568
Sequenzen statt 784) kommt im Einzel-Split auf 79,6 %, also im Rahmen der
Schwankung ähnlich wie die einfache Variante. Der Nutzen des
Rückwärts-Trainings liegt hier weniger in der reinen Accuracy-Zahl als in
der Robustheit gegenüber der Zeichenrichtung im Live-Betrieb.

Bei den Fehlern (siehe `confusion_matrix.png`) verwechselt das Modell vor
allem form- oder bewegungsähnliche Buchstaben, etwa G und O, C und G, N
und W, E und F, A und Y. Das ist bei einer rein trajektorienbasierten
Erkennung nachvollziehbar.

## 7. Verhalten im Live-Betrieb
Im Live-Betrieb läuft dieselbe Preprocessing-Funktion wie im Training
(siehe Abschnitt 4), es gibt also keine zweite, abweichende
Implementierung für den Live-Pfad.

Für das Prüfungsszenario haben wir `record_new_gesture.py` gebaut: das
Skript fragt interaktiv nach Name und Anzahl der neuen Geste, nimmt sie
live auf und trainiert das Modell danach automatisch neu.

Ein Halten der Entf-Taste setzt die aktuell gesammelte Spur zurück, ohne
dass das Programm neu gestartet werden muss (dazu wird dauerhaft "ENTF:
neue Geste" eingeblendet). Ein Retraining wirkt sich allerdings erst nach
einem Neustart von `main.py` aus, ein automatisches Nachladen des Modells
zur Laufzeit gibt es aktuell nicht.

Die gemessenen rund 80 % beziehen sich auf die Hände der fünf
Teammitglieder. Bei einer fremden Hand, etwa der des Prüfers, ist eher mit
einem etwas geringeren Wert zu rechnen.

## 8. Setup zum Nachvollziehen
Umgebung aktivieren:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Danach (Befehle sind auf beiden Systemen gleich):

```bash
# Neue Geste live aufnehmen und Modell automatisch neu trainieren
python record_new_gesture.py

# nur neu trainieren, mit vorhandenem Datensatz
python -c "from GestureRecognition.visualization import evaluate_classifier; evaluate_classifier()"

# Live-Erkennung starten
python main.py
```