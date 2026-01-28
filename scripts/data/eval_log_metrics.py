import json
import re
import sys
from pathlib import Path


def extract_final_answers(log_path: Path):
    answers = []
    for line in log_path.read_text().splitlines():
        marker = "Final Answer:"
        idx = line.rfind(marker)
        if idx != -1:
            answers.append(line[idx + len(marker):].strip())
    return answers


def parse_math_options(options: str):
    pattern = r"([a-e])\s*\)\s*([^,]+)"
    mapping = {}
    for letter, text in re.findall(pattern, options or "", flags=re.IGNORECASE):
        mapping[letter.lower()] = text.strip()
    return mapping


def matches(tokens, answer: str):
    answer_lower = answer.lower()
    return any(token and token.lower() in answer_lower for token in tokens)


def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "mixed_queries.json"
    log_path = root / "evaluation_log.txt"

    if not data_path.exists() or not log_path.exists():
        print("mixed_queries.json or evaluation_log.txt not found.")
        sys.exit(1)

    dataset = json.loads(data_path.read_text())
    answers = extract_final_answers(log_path)

    if len(answers) != len(dataset):
        print(f"[!] Warning: found {len(answers)} answers in log, "
              f"but dataset has {len(dataset)} entries.")

    metrics = {"math": {"correct": 0, "total": 0},
               "science": {"correct": 0, "total": 0}}

    for idx, item in enumerate(dataset):
        if idx >= len(answers):
            break
        final_answer = answers[idx]
        if item.get("type") == "science" or "question" in item:
            metrics["science"]["total"] += 1
            gold_idx = item.get("answer")
            choices = item.get("choices") or []
            gold_tokens = []
            if isinstance(gold_idx, int) and 0 <= gold_idx < len(choices):
                gold_tokens.append(choices[gold_idx])
            elif isinstance(gold_idx, str):
                gold_tokens.append(gold_idx)
            else:
                gold_tokens.extend(choices)
            if matches(gold_tokens, final_answer):
                metrics["science"]["correct"] += 1
        else:
            metrics["math"]["total"] += 1
            correct_letter = (item.get("correct") or "").strip().lower()
            tokens = []
            if correct_letter:
                tokens.extend([correct_letter, f"{correct_letter})", f"{correct_letter} )"])
            option_map = parse_math_options(item.get("options", ""))
            if correct_letter in option_map:
                tokens.append(option_map[correct_letter])
            if matches(tokens, final_answer):
                metrics["math"]["correct"] += 1

    total_correct = metrics["math"]["correct"] + metrics["science"]["correct"]
    total_items = metrics["math"]["total"] + metrics["science"]["total"]

    def fmt(correct, total):
        return f"{correct}/{total} ({(correct / total * 100) if total else 0:.1f}%)" if total else "n/a"

    print("=== Evaluation Summary from Log ===")
    print(f"Math Accuracy: {fmt(metrics['math']['correct'], metrics['math']['total'])}")
    print(f"Science Accuracy: {fmt(metrics['science']['correct'], metrics['science']['total'])}")
    print(f"Overall Accuracy: {fmt(total_correct, total_items)}")


if __name__ == "__main__":
    main()
