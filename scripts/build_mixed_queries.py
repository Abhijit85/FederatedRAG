import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mixed MathQA + ScienceQA evaluation set."
    )
    parser.add_argument(
        "--math-src",
        type=Path,
        default=Path("train_new.json"),
        help="Path to the MathQA source dataset.",
    )
    parser.add_argument(
        "--science-src",
        type=Path,
        default=Path("scienceqa_challenge_test.json"),
        help="Path to the ScienceQA source dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mixed_queries.json"),
        help="Destination path for the mixed dataset.",
    )
    parser.add_argument(
        "--math-count",
        type=int,
        default=10,
        help="Number of MathQA items to sample.",
    )
    parser.add_argument(
        "--science-count",
        type=int,
        default=10,
        help="Number of ScienceQA items to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not args.math_src.exists() or not args.science_src.exists():
        raise FileNotFoundError("Source datasets not found. Check the provided paths.")

    math_data = json.loads(args.math_src.read_text())
    science_data = json.loads(args.science_src.read_text())

    if args.math_count > len(math_data):
        raise ValueError("Requested math_count exceeds available MathQA samples.")
    if args.science_count > len(science_data):
        raise ValueError("Requested science_count exceeds available ScienceQA samples.")

    math_selection = random.sample(math_data, args.math_count)
    science_selection = random.sample(science_data, args.science_count)

    mixed = []
    for item in math_selection:
        mixed.append(
            {
                "type": "math",
                "Problem": item["Problem"],
                "options": item.get("options") or item.get("Options", ""),
                "Rationale": item.get("Rationale", ""),
                "category": item.get("category", "general"),
                "correct": item.get("correct"),
            }
        )

    for item in science_selection:
        science_entry = dict(item)
        science_entry["type"] = "science"
        mixed.append(science_entry)

    args.output.write_text(json.dumps(mixed, indent=2))
    print(
        f"Wrote {len(mixed)} entries "
        f"({args.math_count} math, {args.science_count} science) to {args.output}"
    )


if __name__ == "__main__":
    main()
