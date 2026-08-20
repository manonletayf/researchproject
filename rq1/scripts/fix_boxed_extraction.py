"""
Corrige retroactivement l'extraction 'perturbed_answer' (bug regex
\\boxed{([^}]+)} qui coupe aux accolades imbriquees) en reextrayant
depuis le texte brut deja sauvegarde (full_perturbed_trace /
perturbed_response), sans nouveau calcul GPU.
"""
import json
import argparse


def extract_boxed(text):
    if not text:
        return None
    idx = text.rfind('\\boxed')
    if idx < 0:
        return None
    i = idx + len('\\boxed')
    while i < len(text) and text[i] == ' ':
        i += 1
    if i >= len(text) or text[i] != '{':
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:j].strip()
    return None


args = argparse.ArgumentParser()
args.add_argument('--input_jsonl', type=str, required=True)
args.add_argument('--output_jsonl', type=str, required=True)
args.add_argument('--source_text_field', type=str, default='full_perturbed_trace',
                   help='Champ contenant le texte brut a reextraire (full_perturbed_trace pour S1, perturbed_response pour SoftCoT)')
arg = args.parse_args()

n_total = 0
n_changed = 0
n_still_none = 0

with open(arg.input_jsonl) as fin, open(arg.output_jsonl, 'w') as fout:
    for line in fin:
        d = json.loads(line)
        n_total += 1
        old_answer = d.get('perturbed_answer')
        new_answer = extract_boxed(d.get(arg.source_text_field, ''))
        if new_answer != old_answer:
            n_changed += 1
        if new_answer is None:
            n_still_none += 1
        d['perturbed_answer_old_buggy'] = old_answer
        d['perturbed_answer'] = new_answer
        d['behavioral_sensitivity'] = (str(new_answer) != str(d.get('original_model_answer')))
        fout.write(json.dumps(d) + '\n')

print(f'Total: {n_total}')
print(f'Reponses corrigees (differentes de l ancienne valeur buguee): {n_changed}')
print(f'Toujours None apres correction: {n_still_none}')
