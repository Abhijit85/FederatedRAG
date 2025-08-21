import re
import json

correct_count = 0
with open('mathqa_eval_results_2_step_8b.json','r') as f:
    results=json.load(f)
for r in results:
    r['eval']='wrong'
    if 'predicted' in r:
        match = re.search(r"Final Answer.*?\b([a-e])\b", r["predicted"], re.IGNORECASE | re.DOTALL)
        if match and match.group(1).lower() == r["correct"].lower():
            correct_count += 1
            r['eval']='correct'

with open("mathqa_eval_results_2_step_8b_eval.json", "w") as f:
        json.dump(results, f, indent=2)
accuracy = correct_count / len(results)
print(f"Accuracy: {accuracy:.2%}")
