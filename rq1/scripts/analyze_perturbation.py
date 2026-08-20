"""
Analyse enrichie des trois fichiers de perturbation RQ1 :
- Croisement 2x2 sensibilite x propagation
- Metrique 1 : position de la propagation (proche vs lointaine)
- Metrique 2 : coherence de l'echec (reponse extraite vs None)
- Metrique 3 : similarite texte perturbe vs original (difflib)
- Auditabilite = coverage x propagation_visible (par instance, moyenne par methode)
"""
import json
import difflib
import argparse

args = argparse.ArgumentParser()
args.add_argument('--coconut_jsonl', type=str,
                   default='/home/ubuntu/manon-letayf/rq1_results/coconut_perturb_full.jsonl')
args.add_argument('--softcot_jsonl', type=str,
                   default='/home/ubuntu/manon-letayf/rq1_results/softcot_perturb_50.jsonl')
args.add_argument('--s1_jsonl', type=str,
                   default='/home/ubuntu/manon-letayf/rq1_results/s1_perturb_50.jsonl')
arg = args.parse_args()


def load(path):
    return [json.loads(l) for l in open(path)]


def analyze_2x2(data, sens_key='behavioral_sensitivity', prop_key='propagation_visible'):
    both = sum(1 for d in data if d.get(sens_key) and d.get(prop_key))
    sens_only = sum(1 for d in data if d.get(sens_key) and not d.get(prop_key))
    prop_only = sum(1 for d in data if not d.get(sens_key) and d.get(prop_key))
    neither = sum(1 for d in data if not d.get(sens_key) and not d.get(prop_key))
    return both, sens_only, prop_only, neither


def analyze_answer_coherence(data, answer_key='perturbed_answer'):
    """Metrique 2 : parmi les sensibles, combien ont une reponse extraite (pas None) ?"""
    sensitive = [d for d in data if d.get('behavioral_sensitivity')]
    if not sensitive:
        return None
    coherent = sum(1 for d in sensitive if d.get(answer_key) is not None)
    return coherent, len(sensitive)


def analyze_similarity(data, source_jsonl, trace_key='full_perturbed_trace'):
    """Metrique 3 (corrigee) : similarite entre trace complete perturbee et
    trace complete originale non corrompue (meme instance, meme longueur
    attendue), pas entre l'etape isolee et la trace complete."""
    original_by_id = {}
    with open(source_jsonl) as f:
        for line in f:
            r = json.loads(line)
            original_by_id[r['instance_id']] = r.get('full_trace')

    ratios = []
    for d in data:
        orig_id = d.get('original_instance_id')
        orig_full = original_by_id.get(orig_id)
        trace = d.get(trace_key)
        if orig_full and trace:
            ratio = difflib.SequenceMatcher(None, orig_full, trace).ratio()
            ratios.append(ratio)
    if not ratios:
        return None
    return sum(ratios) / len(ratios), min(ratios), max(ratios)


def analyze_propagation_position(data, corrupt_key, trace_key='full_perturbed_trace'):
    """Metrique 1 : position relative de la 1ere propagation dans la trace (0=debut, 1=fin)."""
    positions = []
    import re
    for d in data:
        if not d.get('propagation_visible'):
            continue
        corrupt_text = d.get(corrupt_key, '')
        trace = d.get(trace_key, '')
        numbers = re.findall(r'\d+', corrupt_text)
        for num in numbers:
            idx = trace.find(num)
            if idx >= 0 and len(trace) > 0:
                positions.append(idx / len(trace))
                break
    if not positions:
        return None
    return sum(positions) / len(positions)


print('=' * 70)
print('COCONUT')
print('=' * 70)
coconut_data = load(arg.coconut_jsonl)
n = len(coconut_data)
print(f'N = {n}')
print('Coverage constant = 0.0 (structurel) -> Auditabilite = 0.0 pour toutes les instances')
print(f'Sensibilite comportementale: {sum(1 for d in coconut_data if d["behavioral_sensitivity"])}/{n}')
print('(Pas de propagation_visible mesurable, pas de 2x2 pertinent)')

print()
print('=' * 70)
print('SOFTCOT')
print('=' * 70)
softcot_data = load(arg.softcot_jsonl)
n = len(softcot_data)
both, sens_only, prop_only, neither = analyze_2x2(softcot_data)
print(f'N = {n}')
print(f'2x2 -- ideal:{both} danger:{sens_only} prop_seule:{prop_only} benin:{neither}')
coh = analyze_answer_coherence(softcot_data)
if coh:
    print(f'Coherence reponse (parmi sensibles): {coh[0]}/{coh[1]} = {coh[0]/coh[1]:.1%} ont une reponse extraite')
sim = analyze_similarity(softcot_data, '/home/ubuntu/manon-letayf/rq1_results/softcot_gsm8k_seed41.jsonl')
if sim:
    print(f'Similarite (difflib) moyenne: {sim[0]:.3f} (min={sim[1]:.3f}, max={sim[2]:.3f})')
pos = analyze_propagation_position(softcot_data, 'corrupted_step_text')
if pos is not None:
    print(f'Position moyenne 1ere propagation: {pos:.2f} (0=debut trace, 1=fin trace)')
coverage_softcot = sum(1 for d in softcot_data if d.get('propagation_visible')) / n
auditability_softcot = coverage_softcot  # coverage proche de 1.0 pour SoftCoT, approx via propagation
print(f'Auditabilite approx (propagation moyenne, proxy coverage~1.0): {auditability_softcot:.3f}')

print()
print('=' * 70)
print('S1')
print('=' * 70)
s1_data = load(arg.s1_jsonl)
n = len(s1_data)
both, sens_only, prop_only, neither = analyze_2x2(s1_data)
print(f'N = {n}')
print(f'2x2 -- ideal:{both} danger:{sens_only} prop_seule:{prop_only} benin:{neither}')
coh = analyze_answer_coherence(s1_data)
if coh:
    print(f'Coherence reponse (parmi sensibles): {coh[0]}/{coh[1]} = {coh[0]/coh[1]:.1%} ont une reponse extraite')
sim = analyze_similarity(s1_data, '/home/ubuntu/manon-letayf/rq1_results/s1_math_batch1.jsonl')
if sim:
    print(f'Similarite (difflib) moyenne: {sim[0]:.3f} (min={sim[1]:.3f}, max={sim[2]:.3f})')
pos = analyze_propagation_position(s1_data, 'corrupted_paragraph_text')
if pos is not None:
    print(f'Position moyenne 1ere propagation: {pos:.2f} (0=debut trace, 1=fin trace)')
auditability_s1 = sum(1 for d in s1_data if d.get('propagation_visible')) / n  # coverage=1.0 exact pour S1
print(f'Auditabilite (coverage=1.0 x propagation): {auditability_s1:.3f}')
