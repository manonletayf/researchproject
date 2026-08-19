"""
Test de sensibilite comportementale par perturbation de pensee latente,
pour le protocole d'injection d'erreur RQ1 (Coconut, §1.10).
Genere une reponse baseline et une reponse perturbee pour chaque instance,
compare, et logge via RunLogger.
"""
import sys
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('/home/ubuntu/manon-letayf/coconut')
sys.path.append('/home/ubuntu/manon-letayf/rq1_common')
from coconut_perturb import CoconutPerturb
from logger import RunLogger

args = argparse.ArgumentParser()
args.add_argument('--checkpoint', type=str,
                   default='/home/ubuntu/manon-letayf/coconut/checkpoints/gsm_coconut/gsm-coconut/checkpoint_9')
args.add_argument('--data_path', type=str, default='data/gsm_test.json')
args.add_argument('--model_id', type=str, default='openai-community/gpt2')
args.add_argument('--max_latent_stage', type=int, default=3)
args.add_argument('--c_thought', type=int, default=2)
args.add_argument('--n_instances', type=int, default=50)
args.add_argument('--perturb_mode', type=str, default='noise', choices=['noise', 'zero'])
args.add_argument('--perturb_target', type=str, default='latent', choices=['latent', 'question_control'],
                   help='latent: perturbe une pensee latente (defaut). question_control: perturbe un token de la question (test de controle pour isoler la fragilite du greedy decoding)')
args.add_argument('--perturb_scale', type=float, default=1.0)
args.add_argument('--run_id', type=str, required=True)
args.add_argument('--output_jsonl', type=str, required=True)
arg = args.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Chargement du tokenizer et du modele...')
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
model = CoconutPerturb(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)

saved_weights = torch.load(arg.checkpoint, map_location=device)
model.load_state_dict(saved_weights, strict=False)
model.to(device)
model.eval()

k = arg.max_latent_stage * arg.c_thought
mid_pass_idx = k // 2  # position latente intermediaire a perturber

print(f'k={k} tokens latents par instance, perturbation a pass_idx={mid_pass_idx}')

data = json.load(open(arg.data_path))[:arg.n_instances]
run_logger = RunLogger(arg.output_jsonl)

for i, sample in enumerate(data):
    instance_id = f'coconut_perturb_test_{i:05d}'
    if run_logger.is_done(instance_id):
        continue

    question_tokenized = tokenizer.encode(sample['question'] + '\n', add_special_tokens=True)
    tokens = question_tokenized + [start_id] + [latent_id] * k + [end_id]
    input_ids = torch.tensor(tokens).view(1, -1).to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out_baseline = model.generate_perturbed(input_ids, attention_mask, max_new_tokens=64,
                                                  perturb_pass_idx=None)
        if arg.perturb_target == 'latent':
            out_perturbed = model.generate_perturbed(input_ids, attention_mask, max_new_tokens=64,
                                                       perturb_pass_idx=mid_pass_idx,
                                                       perturb_mode=arg.perturb_mode,
                                                       perturb_scale=arg.perturb_scale)
        else:
            question_len = len(question_tokenized)
            control_idx = question_len // 2
            perturbed_embeds = model.embedding(input_ids).clone()
            vec = perturbed_embeds[0, control_idx, :]
            if arg.perturb_mode == 'noise':
                noise = torch.randn_like(vec) * vec.norm() * arg.perturb_scale
                perturbed_embeds[0, control_idx, :] = vec + noise
            else:
                perturbed_embeds[0, control_idx, :] = torch.zeros_like(vec)
            out_perturbed = model.generate_perturbed_from_embeds(perturbed_embeds, attention_mask, max_new_tokens=64)

    text_baseline = tokenizer.decode(out_baseline[0], skip_special_tokens=False)
    text_perturbed = tokenizer.decode(out_perturbed[0], skip_special_tokens=False)

    def extract_answer(text):
        if '###' in text:
            return text.split('###')[-1].strip()
        return None

    answer_baseline = extract_answer(text_baseline)
    answer_perturbed = extract_answer(text_perturbed)
    gold_answer = sample['answer']

    behavioral_sensitivity = (answer_baseline != answer_perturbed)

    run_logger.log({
        'instance_id': instance_id,
        'run_id': arg.run_id,
        'method': 'coconut',
        'task': 'gsm8k',
        'gold_answer': gold_answer,
        'answer_baseline': answer_baseline,
        'answer_perturbed': answer_perturbed,
        'baseline_correct': (answer_baseline == gold_answer),
        'behavioral_sensitivity': behavioral_sensitivity,
        'perturb_pass_idx': mid_pass_idx,
        'perturb_mode': arg.perturb_mode,
        'perturb_scale': arg.perturb_scale,
        'k_latent_tokens': k,
    })

    if (i + 1) % 10 == 0:
        print(f'{i+1}/{len(data)} traitees')

print('Termine.')
