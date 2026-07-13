import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GestureRecognition"))

from labeling import data_labeling

def main():
    parser = argparse.ArgumentParser(
        description="Neue Geste live aufnehmen und Modell neu trainieren (Pruefungsszenario)."
    )
    parser.add_argument("label", nargs="?", default=None, help="Name der neuen Geste/Klasse, z.B. 'Kreis'")
    parser.add_argument("--times", type=int, default=None, help="Anzahl Aufnahmen (Standard: 10)")
    parser.add_argument("--no-retrain", action="store_true", help="Nach der Aufnahme nicht automatisch neu trainieren")
    args = parser.parse_args()

    if not args.label:
        args.label = input("Name der neuen Geste: ").strip()
    if not args.times:
        times = input("Wie oft aufnehmen? [10]: ").strip()
        args.times = int(times) if times else 10

    print(f"Neue Geste '{args.label}': {args.times} Aufnahmen werden aufgenommen.\n")
    data_labeling(times=args.times, label=args.label)

    if args.no_retrain:
        print("\n--no-retrain gesetzt: Modell wurde nicht neu trainiert.")
        return

    print(f"\nTrainiere Modell neu mit '{args.label}' und allen bisherigen Klassen ...\n")
    from GestureRecognition.visualization import evaluate_classifier
    evaluate_classifier()

    print("\nFertig! Live testen mit: python main.py")

if __name__ == "__main__":
    main()
