"""
Evaluation Coconut avec logging par instance (RunLogger), format
coherent avec S1/SoftCoT pour RQ1 - accuracy, gold_plan_steps,
reasoning_units.
"""
import sys
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('/home/ubuntu/manon-letayf/coconut')
sys.path.append('/home/ubuntu/manon-letayf/rq1_common')
from coconut import Coconut
from logger import RunLogger

args = argparse.ArgumentParser()
args.add_argument('--checkpoint', type=str,
                   default='/home/ubuntu/manon-letayf/coconut/checkpoints/gsm_coconut/gsm-coconut/checkpoint_9')
args.add_argument('--data_path', type=str, default='data/gsm_test.json')
args.add_argument('--model_id', type=str, default='openai-community/gpt2')
args.add_argument('--max_latent_stage', type=int, default=3)
args.add_argument('--c_thought', type=int, default=2)
args.add_argument('--n_instances', type=int, default=500)
args.add_argument('--run_id', type=str, required=True)
args.add_argument('--output_jsonl', type=str, required=True)
arg = args.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(arg.model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens('<|start-latent|>')
tokenizer.add_tokens('<|end-latent|>')
tokenizer.add_tokens('<|latent|>')
latent_id = tokenizer.convert_tokens_to_ids('<|latent|>')
start_id = tokenizer.convert_tokens_to_ids('<|start-latent|>')
end_id = tokenizer.convert_tokens_to_ids('<|end-latent|>')

base_model = AutoModelForCausalLM.from_pretrained(arg.model_id)
base_model.resize_token_embeddings(len(tokenizer))
model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)

saved_weights = torch.load(arg.checkpoint, map_location=device)
result = model.load_state_dict(saved_weights, strict=False)
assert not result.missing_keys and not result.unexpected_keys, \
    f"Chargement incomplet: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
model.to(device)
model.eval()

k = arg.max_latent_stage * arg.c_thought

data = json.load(open(arg.data_path))[:arg.n_instances]
run_logger = RunLogger(arg.output_jsonl)

correct_count = 0
for i, sample in enumerate(data):
    instance_id = f'coconut_gsm8k_test_{i:05d}'
    if run_logger.is_done(instance_id):
        continue

    question_tokenized = tokenizer.encode(sample['question'] + '\n', add_special_tokens=True)
    tokens = question_tokenized + [start_id] + [latent_id] * k + [end_id]
    input_ids = torch.tensor(tokens).view(1, -1).to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model.generate(input_ids, attention_mask, max_new_tokens=64)

    text = tokenizer.decode(out[0], skip_special_tokens=False)
    model_answer = text.split('###')[-1].strip() if '###' in text else None
    gold_answer = sample['answer']
    is_correct = (model_answer == gold_answer)
    if is_correct:
        correct_count += 1

    gold_plan_steps = len(sample.get('steps', []))
    reasoning_units = min(gold_plan_steps, arg.max_latent_stage) * arg.c_thought

    run_logger.log({
        'instance_id': instance_id,
        'run_id': arg.run_id,
        'method': 'coconut',
        'task': 'gsm8k',
        'gold_answer': gold_answer,
        'gold_plan_steps': gold_plan_steps,
        'model_answer': model_answer,
        'correct': bool(is_correct),
        'full_trace': text,
        'reasoning_units': reasoning_units,
        'k_latent_tokens': k,
        'prefix_token_count': k,
        'coverage': 0.0,
    })

    if (i + 1) % 50 == 0:
        print(f'{i+1}/{len(data)} - accuracy so far: {correct_count/(i+1):.1%}')

print(f'Termine. Accuracy finale: {correct_count}/{len(data)} = {correct_count/len(data):.1%}')
