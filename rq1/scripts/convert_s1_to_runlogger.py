"""
Script de conversion : lit les fichiers samples_*.jsonl produits par
lm_eval (--log_samples) et les convertit dans le schéma RunLogger
commun de RQ1 (thinking_token_count, wait_insertions, gold_plan_steps,
coverage, has_thinking_block).
"""
import sys
import json
import argparse
import re

sys.path.append('/home/ubuntu/manon-letayf/rq1_common')
from logger import RunLogger

from transformers import AutoTokenizer

THINK_START = "<|im_start|>think"
THINK_END = "<|im_start|>answer"

args = argparse.ArgumentParser()
args.add_argument('--input_jsonl', type=str, required=True,
                   help='Fichier samples_*.jsonl produit par lm_eval --log_samples')
args.add_argument('--task_name', type=str, required=True,
                   help='Nom de la tache lm_eval (ex: openai_math_batch1, aime24_nofigures, gpqa_diamond_openai)')
args.add_argument('--task_family', type=str, required=True, choices=['math', 'aime', 'gpqa'],
                   help='Famille de tache, determine comment calculer gold_plan_steps')
args.add_argument('--run_id', type=str, required=True)
args.add_argument('--output_jsonl', type=str, required=True)
args.add_argument('--backbone', type=str, default='simplescaling/s1.1-32B')
args.add_argument('--tokenizer_id', type=str, default='simplescaling/s1-32B')
args.add_argument('--is_baseline_no_cot', action='store_true', default=False)
args.add_argument('--instance_offset', type=int, default=0,
                   help='Offset pour eviter les collisions instance_id entre batches')
args.add_argument('--thinking_n_ignore', type=int, required=True, help='Valeur de thinking_n_ignore utilisee pour ce run (config connue, pas devinee)')
args.add_argument('--wait_str', type=str, default='Wait',
                   help='Chaine d insertion du budget forcing a compter')
arg = args.parse_args()

print(f'Chargement du tokenizer {arg.tokenizer_id}...')
tokenizer = AutoTokenizer.from_pretrained(arg.tokenizer_id)

run_logger = RunLogger(arg.output_jsonl)

n_processed = 0
n_skipped = 0
n_no_thinking = 0

with open(arg.input_jsonl, 'r') as f:
    for line_idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)

        doc_id = data.get('doc_id', line_idx)
        instance_id = f'{arg.task_name}_doc{arg.instance_offset + doc_id:05d}'

        if run_logger.is_done(instance_id):
            n_skipped += 1
            continue

        doc = data.get('doc', {})
        resps = data.get('filtered_resps', data.get('resps', [None]))
        full_trace = resps[0] if resps else None
        if isinstance(full_trace, list):
            full_trace = full_trace[0] if full_trace else None

        correct = bool(data.get('exact_match', 0))
        model_answer = None
        extracted = data.get('extracted_answers')
        if extracted:
            model_answer = extracted[0] if isinstance(extracted, list) else extracted

        gold_answer = data.get('target', doc.get('answer'))

        # --- has_thinking_block / thinking_token_count / wait_insertions ---
        has_thinking_block = False
        thinking_token_count = None
        wait_insertions = 0
        if full_trace is not None:
            wait_occurrences_total = full_trace.count(arg.wait_str)
            if THINK_START in full_trace:
                has_thinking_block = True
                after_start = full_trace.split(THINK_START, 1)[1]
                if THINK_END in after_start:
                    thinking_part = after_start.split(THINK_END, 1)[0]
                else:
                    thinking_part = after_start
                thinking_token_count = len(tokenizer.encode(thinking_part))
            else:
                n_no_thinking += 1

        # --- coverage ---
        coverage = 1.0 if has_thinking_block else None

        # --- trace_token_count (longueur totale) ---
        trace_token_count = len(tokenizer.encode(full_trace)) if full_trace else None

        # --- gold_plan_steps selon la famille de tache ---
        gold_plan_steps = None
        if arg.task_family == 'math':
            solution = doc.get('solution')
            if solution:
                gold_plan_steps = len([l for l in solution.split('\n') if l.strip()])
        elif arg.task_family == 'gpqa':
            explanation = doc.get('Explanation') or doc.get('explanation')
            if explanation:
                gold_plan_steps = len([s for s in re.split(r'[.!?]+', explanation) if s.strip()])
        elif arg.task_family == 'aime':
            gold_plan_steps = None  # pas de solution detaillee dans le dataset AIME

        record = {
            'instance_id': instance_id,
            'run_id': arg.run_id,
            'method': 's1',
            'task': arg.task_name,
            'backbone': arg.backbone,
            'is_baseline_no_cot': arg.is_baseline_no_cot,
            'gold_answer': gold_answer,
            'gold_plan_steps': gold_plan_steps,
            'model_answer': model_answer,
            'correct': correct,
            'full_trace': full_trace,
            'trace_token_count': trace_token_count,
            'prefix_token_count': 0,
            'thinking_token_count': thinking_token_count,
            'wait_insertions_forced': arg.thinking_n_ignore,
            'wait_occurrences_total': wait_occurrences_total,
            'has_thinking_block': has_thinking_block,
            'coverage': coverage,
            'latency_s': None,
        }
        run_logger.log(record)
        n_processed += 1

print(f'Termine. {n_processed} instances converties, {n_skipped} deja presentes (sautees), {n_no_thinking} sans bloc de reflexion detecte.')
