import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GestureRecognition"))

from labeling import data_labeling

GESTURES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
PER_PERSON = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nimmt Gesten auf. Ohne Angabe: komplettes Alphabet A-Z mit je 6 Aufnahmen."
    )
    parser.add_argument(
        "gestures", nargs="*",
        help="Gezielt einzelne Gesten nachnehmen, z.B. 'Y:2 V:1' fuer 2x Y und 1x V.",
    )
    return parser.parse_args()


def build_plan(args):
    if not args.gestures:
        return [(g, PER_PERSON) for g in GESTURES]

    plan = []
    for item in args.gestures:
        label, _, count = item.partition(":")
        plan.append((label.upper(), int(count) if count else PER_PERSON))
    return plan


args = parse_args()
plan = build_plan(args)

print(f"Aufnahme-Plan: {', '.join(f'{label}:{times}' for label, times in plan)}\n")

for gesture, times in plan:
    print(f"\n{'='*40}")
    print(f"  Geste: {gesture}  --  {times} Aufnahmen")
    print(f"{'='*40}")
    data_labeling(times=times, label=gesture)

print("\nFertig!")
