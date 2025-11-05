import os
import json
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
import pandas as pd
import platformdirs
import textgrad as tg
from .base import Dataset

# The below metric is taken from DSPy for consistenc
# and modified to work with TG-graphs

def parse_integer_answer(answer: str, only_first_line: bool=False):
    try:
        if only_first_line:
            answer = answer.strip().split('\n')[0]
        answer = answer.strip()
        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split('.')[0]
        answer = ''.join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0
    
    return answer

def string_based_equality_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    return int(parse_integer_answer(str(prediction.value)) == int(parse_integer_answer(str(ground_truth_answer.value))))


class BigBenchHard(Dataset):
    def __init__(self, task_name: str, root: str=None, split: str="train", *args, **kwargs):
        """
        Tasks from BIG-Bench Hard
        <https://github.com/suzgunmirac/BIG-Bench-Hard>

        The train, val, test splits were constructed from  50/100/100 examples.

        Args:
            root (string): Root directory of the dataset
            split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"`` and ``"test"``.
        """
        if root is None:
            root = _dataset_root()
        self.root = root
        self.split = split
        self.task_name = task_name
        self._check_or_download_dataset()
        assert split in ["train", "val", "test"]
        data_path = os.path.join(self.root, self.task_name, f"{split}.csv")
        self.data = pd.read_csv(data_path, index_col=0)
        self._task_description = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
    
    def get_task_description(self):
        return self._task_description 
       
    def _resolve_dataset_json(self) -> Optional[str]:
        root_json = os.path.join(self.root, f"{self.task_name}.json")
        if os.path.exists(root_json):
            return root_json
        packaged = DATA_DIR / f"{self.task_name}.json"
        if packaged.exists():
            return str(packaged)
        return None

    def _download_dataset(self) -> str:
        target = os.path.join(self.root, f"{self.task_name}.json")
        url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{self.task_name}.json"
        try:
            urlretrieve(url, target)
        except Exception as exc:  # pragma: no cover - network required
            raise RuntimeError(
                f"Failed to download BIG-Bench-Hard task '{self.task_name}' from {url}. "
                "Provide the dataset locally via TEXTGRAD_CACHE_DIR or tasks/data."
            ) from exc
        return target

    def _check_or_download_dataset(self):
        split_path = os.path.join(self.root, self.task_name, f"{self.split}.csv")
        if os.path.exists(split_path):
            return

        os.makedirs(os.path.join(self.root, self.task_name), exist_ok=True)

        dataset_json = self._resolve_dataset_json()
        if dataset_json is None:
            dataset_json = self._download_dataset()

        with open(dataset_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        examples = data["examples"]
        train_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[:50]]
        val_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[50:150]]
        test_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[150:]]
        train_path = os.path.join(self.root, self.task_name, "train.csv")
        with open(train_path, "w") as f:
            pd.DataFrame(train_examples).to_csv(f)
        val_path = os.path.join(self.root, self.task_name, "val.csv")
        with open(val_path, "w") as f:
            pd.DataFrame(val_examples).to_csv(f)
        test_path = os.path.join(self.root, self.task_name, "test.csv")
        with open(test_path, "w") as f:
            pd.DataFrame(test_examples).to_csv(f)
        
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row["x"], row["y"]
    
    def __len__(self):
        return len(self.data)
    
    def get_default_task_instruction(self):
        return self._task_description
    
DATA_DIR = Path(__file__).resolve().parent / "data"


def _dataset_root() -> str:
    override = os.getenv("TEXTGRAD_CACHE_DIR")
    if override:
        os.makedirs(override, exist_ok=True)
        return override
    default_root = platformdirs.user_cache_dir("textgrad")
    try:
        os.makedirs(default_root, exist_ok=True)
        return default_root
    except OSError:
        fallback = os.path.join(os.getcwd(), ".cache", "textgrad")
        os.makedirs(fallback, exist_ok=True)
        return fallback
