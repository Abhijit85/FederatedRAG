import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-client evaluation datasets with IID or non-IID distributions."
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=2,
        help="Number of synthetic clients to generate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["math", "science"],
        default=["math", "science"],
        help="Datasets to include for each client.",
    )
    parser.add_argument(
        "--distribution",
        choices=["iid", "noniid"],
        default="iid",
        help="Sampling mode for client datasets.",
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
        "--math-per-client",
        type=int,
        default=10,
        help="Number of MathQA items per client dataset.",
    )
    parser.add_argument(
        "--science-per-client",
        type=int,
        default=10,
        help="Number of ScienceQA items per client dataset.",
    )
    parser.add_argument(
        "--math-categories",
        type=str,
        help="Comma-separated list of MathQA categories to use for non-IID splits.",
    )
    parser.add_argument(
        "--science-topics",
        type=str,
        help="Comma-separated list of ScienceQA topics/categories to use for non-IID splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("client_datasets"),
        help="Directory where client datasets will be stored.",
    )
    return parser.parse_args()


def _parse_list(option_value: Optional[str]) -> List[str]:
    if not option_value:
        return []
    return [item.strip() for item in option_value.split(",") if item.strip()]


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict) and "examples" in payload and isinstance(payload["examples"], list):
        return payload["examples"]
    return payload


def _ensure_capacity(total_needed: int, available: int, label: str):
    if total_needed > available:
        raise ValueError(f"Requested {total_needed} {label} samples but only {available} are available.")


def _iid_splits(
    data: Sequence[Dict[str, object]],
    clients: int,
    per_client: int,
) -> List[List[Dict[str, object]]]:
    if per_client == 0:
        return [[] for _ in range(clients)]
    _ensure_capacity(clients * per_client, len(data), "IID")
    shuffled = list(data)
    random.shuffle(shuffled)
    splits: List[List[Dict[str, object]]] = []
    index = 0
    for _ in range(clients):
        splits.append(shuffled[index:index + per_client])
        index += per_client
    return splits


def _assign_categories(
    pools: Dict[str, List[Dict[str, object]]],
    clients: int,
    per_client: int,
    preferred: List[str],
    dataset_label: str,
) -> List[str]:
    if per_client == 0:
        return ["n/a"] * clients
    if not pools:
        raise ValueError(f"No categorical information available for {dataset_label} dataset.")

    # Determine which category each client will receive.
    available_categories = [cat for cat, docs in pools.items() if docs]
    if not available_categories:
        raise ValueError(f"No data available to create {dataset_label} non-IID splits.")

    if preferred:
        chosen_sequence = []
        for idx in range(clients):
            category = preferred[idx % len(preferred)].lower()
            if category not in pools:
                raise ValueError(
                    f"Preferred {dataset_label} category '{category}' not found in the source dataset."
                )
            chosen_sequence.append(category)
    else:
        # Sort categories by availability to keep splits feasible.
        sorted_categories = [cat for cat, _ in Counter({cat: len(docs) for cat, docs in pools.items()}).most_common()]
        chosen_sequence = [sorted_categories[idx % len(sorted_categories)] for idx in range(clients)]
    return chosen_sequence


def _noniid_splits(
    data: Sequence[Dict[str, object]],
    clients: int,
    per_client: int,
    key_candidates: Sequence[str],
    preferred_categories: List[str],
    dataset_label: str,
) -> tuple[List[List[Dict[str, object]]], List[str]]:
    if per_client == 0:
        return [[] for _ in range(clients)], ["n/a"] * clients

    def _extract_key(item: Dict[str, object]) -> str:
        for key in key_candidates:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return "unknown"

    pools: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in data:
        pools[_extract_key(record)].append(record)

    for doc_list in pools.values():
        random.shuffle(doc_list)

    selection_sequence = _assign_categories(pools, clients, per_client, preferred_categories, dataset_label)

    splits: List[List[Dict[str, object]]] = []
    for category in selection_sequence:
        pool = pools.get(category, [])
        if len(pool) < per_client:
            raise ValueError(
                f"Category '{category}' in {dataset_label} dataset does not have enough samples "
                f"({len(pool)} available, {per_client} required)."
            )
        take = [pool.pop() for _ in range(per_client)]
        splits.append(take)

    return splits, selection_sequence


def _transform_math_entry(item: Dict[str, object]) -> Dict[str, object]:
    problem = item.get("Problem") or item.get("problem")
    options = item.get("options") or item.get("Options", "")
    rationale = item.get("Rationale", "")
    correct = item.get("correct") or item.get("Answer")

    if not problem and item.get("input"):
        problem = item.get("input")
    if not correct and item.get("target"):
        correct = item.get("target")

    return {
        "type": "math",
        "Problem": problem,
        "options": options,
        "Rationale": rationale,
        "category": item.get("category") or item.get("Category", "general"),
        "correct": correct,
    }


def _transform_science_entry(item: Dict[str, object]) -> Dict[str, object]:
    entry = dict(item)
    entry["type"] = "science"
    return entry


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.clients < 1:
        raise ValueError("Number of clients must be at least 1.")

    include_math = "math" in args.datasets
    include_science = "science" in args.datasets

    math_data = _load_json(args.math_src) if include_math else []
    science_data = _load_json(args.science_src) if include_science else []

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    math_splits: List[List[Dict[str, object]]] = [[] for _ in range(args.clients)]
    math_focus: List[str] = ["n/a"] * args.clients
    if include_math:
        if args.distribution == "iid":
            math_splits = _iid_splits(math_data, args.clients, args.math_per_client)
        else:
            math_categories = _parse_list(args.math_categories)
            math_splits, math_focus = _noniid_splits(
                math_data,
                args.clients,
                args.math_per_client,
                key_candidates=["category", "Category"],
                preferred_categories=math_categories,
                dataset_label="MathQA",
            )

    science_splits: List[List[Dict[str, object]]] = [[] for _ in range(args.clients)]
    science_focus: List[str] = ["n/a"] * args.clients
    if include_science:
        if args.distribution == "iid":
            science_splits = _iid_splits(science_data, args.clients, args.science_per_client)
        else:
            science_topics = _parse_list(args.science_topics)
            science_splits, science_focus = _noniid_splits(
                science_data,
                args.clients,
                args.science_per_client,
                key_candidates=["topic", "category", "subject"],
                preferred_categories=science_topics,
                dataset_label="ScienceQA",
            )

    summary: List[Dict[str, object]] = []

    for index in range(args.clients):
        client_entries: List[Dict[str, object]] = []

        for item in math_splits[index]:
            client_entries.append(_transform_math_entry(item))

        for item in science_splits[index]:
            client_entries.append(_transform_science_entry(item))

        random.shuffle(client_entries)

        dataset_path = output_dir / f"client_{index + 1}_mixed_queries.json"
        dataset_path.write_text(json.dumps(client_entries, indent=2), encoding="utf-8")

        summary.append(
            {
                "client": index + 1,
                "dataset_path": str(dataset_path),
                "math_count": len(math_splits[index]),
                "science_count": len(science_splits[index]),
                "math_focus": math_focus[index],
                "science_focus": science_focus[index],
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[✓] Generated datasets for {args.clients} clients in '{output_dir}'.")
    for entry in summary:
        math_note = f"math={entry['math_count']} (focus: {entry['math_focus']})" if include_math else "math=0"
        science_note = (
            f"science={entry['science_count']} (focus: {entry['science_focus']})" if include_science else "science=0"
        )
        print(f"  - Client {entry['client']}: {math_note}, {science_note} → {entry['dataset_path']}")


if __name__ == "__main__":
    main()
