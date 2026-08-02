#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mongo_utils import MongoVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Atlas Vector Search index for the MathQA Mongo collection.")
    parser.add_argument("--db-name", type=str, default="FredRag")
    parser.add_argument("--collection", type=str, default=os.environ.get("MATHQA_COLLECTION", "math_problems"))
    parser.add_argument("--index-name", type=str, default=os.environ.get("MONGO_VECTOR_INDEX_NAME", "vector_index"))
    parser.add_argument("--dimensions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI must be set.")

    args = parse_args()
    os.environ["MONGO_VECTOR_INDEX_NAME"] = args.index_name
    store = MongoVectorStore(mongo_uri, args.db_name, args.collection)
    result = store.ensure_vector_search_index(dimensions=args.dimensions, index_name=args.index_name)
    print(result)


if __name__ == "__main__":
    main()
