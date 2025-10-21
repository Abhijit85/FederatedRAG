import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mixed MathQA + ScienceQA evaluation set."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["math", "science"],
        default=["math", "science"],
        help=(
            "Datasets to include in the mixed output. "
            "Choose one or both of: 'math', 'science'."
        ),
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
        "--distribution",
        choices=["iid", "noniid"],
        default="iid",
        help=(
            "Sampling strategy. 'iid' draws uniformly across datasets. "
            "'noniid' restricts each selected dataset to a single dominant category/topic."
        ),
    )
    parser.add_argument(
        "--math-category",
        type=str,
        help="Category name to target for MathQA when using 'noniid' distribution.",
    )
    parser.add_argument(
        "--science-topic",
        type=str,
        help="Topic/category name to target for ScienceQA when using 'noniid' distribution.",
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

    selected_datasets = set(args.datasets)
    include_math = "math" in selected_datasets
    include_science = "science" in selected_datasets

    if include_math and not args.math_src.exists():
        raise FileNotFoundError(f"Math dataset not found at {args.math_src}.")
    if include_science and not args.science_src.exists():
        raise FileNotFoundError(f"Science dataset not found at {args.science_src}.")

    math_data = json.loads(args.math_src.read_text()) if include_math else []
    science_data = json.loads(args.science_src.read_text()) if include_science else []

    math_count = args.math_count if include_math else 0
    science_count = args.science_count if include_science else 0

    if not include_math and args.math_count:
        print("⚠️ Ignoring --math-count because 'math' dataset was not selected.")
    if not include_science and args.science_count:
        print("⚠️ Ignoring --science-count because 'science' dataset was not selected.")

    if include_math and math_count > len(math_data):
        raise ValueError("Requested math_count exceeds available MathQA samples.")
    if include_science and science_count > len(science_data):
        raise ValueError("Requested science_count exceeds available ScienceQA samples.")

    def _select_noniid_subset(
        data,
        count,
        key_candidates,
        preferred_value,
        dataset_label,
    ):
        if count == 0:
            return [], None

        # Pick the first key that exists in the records.
        chosen_key = None
        for key in key_candidates:
            if any(key in item for item in data):
                chosen_key = key
                break
        if not chosen_key:
            raise ValueError(
                f"Could not find a descriptive field ({', '.join(key_candidates)}) "
                f"in {dataset_label} dataset to construct a non-IID slice."
            )

        def _normalize(value):
            return (value or "").strip().lower()

        if preferred_value:
            target = preferred_value.strip().lower()
            filtered = [item for item in data if _normalize(item.get(chosen_key, "")) == target]
            if len(filtered) < count:
                raise ValueError(
                    f"Requested {count} {dataset_label} samples from category '{preferred_value}', "
                    f"but only found {len(filtered)}."
                )
            return random.sample(filtered, count), preferred_value

        category_counts = Counter(_normalize(item.get(chosen_key, "unknown")) for item in data)
        dominant_value, _ = category_counts.most_common(1)[0]
        filtered = [item for item in data if _normalize(item.get(chosen_key, "")) == dominant_value]
        if len(filtered) < count:
            raise ValueError(
                f"Dominant category '{dominant_value}' in {dataset_label} dataset does not have "
                f"enough samples ({len(filtered)} < {count})."
            )
        return random.sample(filtered, count), dominant_value

    math_selection = []
    science_selection = []
    math_focus = None
    science_focus = None

    if include_math and math_count:
        if args.distribution == "iid":
            math_selection = random.sample(math_data, math_count)
        else:
            math_selection, math_focus = _select_noniid_subset(
                math_data,
                math_count,
                key_candidates=["category", "Category"],
                preferred_value=args.math_category,
                dataset_label="MathQA",
            )

    if include_science and science_count:
        if args.distribution == "iid":
            science_selection = random.sample(science_data, science_count)
        else:
            science_selection, science_focus = _select_noniid_subset(
                science_data,
                science_count,
                key_candidates=["topic", "category", "subject"],
                preferred_value=args.science_topic,
                dataset_label="ScienceQA",
            )

    mixed = []
    if include_math:
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

    if include_science:
        for item in science_selection:
            science_entry = dict(item)
            science_entry["type"] = "science"
            mixed.append(science_entry)

    if not mixed:
        raise ValueError("No datasets selected; nothing to write.")

    args.output.write_text(json.dumps(mixed, indent=2))

    focus_msg = []
    if args.distribution == "noniid":
        if math_focus:
            focus_msg.append(f"Math focus: {math_focus}")
        if science_focus:
            focus_msg.append(f"Science focus: {science_focus}")
    focus_suffix = f" ({'; '.join(focus_msg)})" if focus_msg else ""

    print(
        f"Wrote {len(mixed)} entries "
        f"({math_count} math, {science_count} science) to {args.output}{focus_suffix}"
    )


if __name__ == "__main__":
    main()
