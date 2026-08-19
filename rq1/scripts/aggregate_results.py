"""
Agrege les fichiers JSONL de resultats RQ1 (par methode/tache) et
calcule les statistiques de base (accuracy, overthinking moyen).
"""
import json
import argparse
from collections import defaultdict

args = argparse.ArgumentParser()
args.add_argument('--input_jsonl', type=str, nargs='+', required=True,
                   help='Un ou plusieurs fichiers JSONL a agreger')
args.add_argument('--group_by', type=str, default='task',
                   help='Champ pour grouper les stats (ex: task, method)')
arg = args.parse_args()

groups = defaultdict(list)

for path in arg.input_jsonl:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = d.get(arg.group_by, 'unknown')
            groups[key].append(d)

print(f"{'Groupe':<30} {'N':>6} {'Correct':>8} {'Accuracy':>10}")
print('-' * 60)
total_n = 0
total_correct = 0
for key, records in sorted(groups.items()):
    n = len(records)
    correct = sum(1 for r in records if r.get('correct'))
    acc = correct / n if n > 0 else 0
    print(f"{key:<30} {n:>6} {correct:>8} {acc:>10.1%}")
    total_n += n
    total_correct += correct

print('-' * 60)
overall_acc = total_correct / total_n if total_n > 0 else 0
print(f"{'TOTAL':<30} {total_n:>6} {total_correct:>8} {overall_acc:>10.1%}")

# Overthinking moyen si les champs existent
thinking_tokens = [r['thinking_token_count'] for records in groups.values()
                    for r in records if r.get('thinking_token_count') is not None]
if thinking_tokens:
    print(f"\nthinking_token_count moyen: {sum(thinking_tokens)/len(thinking_tokens):.1f}")

wait_forced = [r['wait_insertions_forced'] for records in groups.values()
               for r in records if r.get('wait_insertions_forced') is not None]
if wait_forced:
    print(f"wait_insertions_forced moyen: {sum(wait_forced)/len(wait_forced):.2f}")
