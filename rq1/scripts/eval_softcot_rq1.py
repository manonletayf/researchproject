"""
Wrapper autour de evaluate_softcot.py (logique originale des auteurs
préservée à l'identique) qui branche notre RunLogger + wandb pour
produire des données exploitables pour RQ1 et un suivi en temps réel.
"""
import re
import sys
import argparse
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from fastNLP import logger
import wandb

sys.path.append('/home/ubuntu/manon-letayf/rq1_common')
from logger import RunLogger

from llm_model import EfficientSoftCoTFromSmallModel
from data_loader import GSM8KLoader, StrategyQALoader, AugASDivLoader, AQuALoader, DULoader
from utils import pre_process_gsm8k, pre_process_strategy_qa, pre_process_aqua, pre_process_du


args = argparse.ArgumentParser()
args.add_argument('--base_model_id', type=str, default='Qwen/Qwen2.5-7B-Instruct')
args.add_argument('--assistant_model_id', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
args.add_argument('--params_file_name', type=str, default=None)
args.add_argument('--base_model_ckpt', type=str, default=None)
args.add_argument('--assistant_model_ckpt', type=str, default=None)
args.add_argument('--num_thought_tokens', type=int, default=4)
args.add_argument('--num_return_sequences', type=int, default=1)
args.add_argument('--task_name', type=str, choices=['gsm8k', 'strategyqa', 'asdiv-aug', 'aqua', 'du'])
args.add_argument('--print_input', action='store_true', default=False)
args.add_argument('--print_response', action='store_true', default=False)
args.add_argument('--test_k', type=int, default=0)
args.add_argument('--seed', type=int, default=42)
args.add_argument('--tune_base_model', action='store_true', default=False)
args.add_argument('--tune_assistant_model', action='store_true', default=False)
args.add_argument('--run_id', type=str, required=True, help='Identifiant du run, ex: softcot-gsm8k-seed41')
args.add_argument('--output_jsonl', type=str, required=True, help='Chemin du fichier JSONL de sortie')
args.add_argument('--is_baseline_no_cot', action='store_true', default=False)
args.add_argument('--no_wandb', action='store_true', default=False, help='Desactive le logging wandb')
arg = args.parse_args()
logger.info(f'Args: {arg.__dict__}')

base_model_id = arg.base_model_id
assistant_model_id = arg.assistant_model_id
params_file_name = arg.params_file_name
base_model_ckpt = arg.base_model_ckpt
assistant_model_ckpt = arg.assistant_model_ckpt
num_thought_tokens = arg.num_thought_tokens
num_return_sequences = arg.num_return_sequences
task_name = arg.task_name
print_input = arg.print_input
print_response = arg.print_response
test_k = arg.test_k
seed = arg.seed
tune_base_model = arg.tune_base_model
tune_assistant_model = arg.tune_assistant_model

run_logger = RunLogger(arg.output_jsonl)

use_wandb = not arg.no_wandb
if use_wandb:
    wandb.init(
        project="coconut",
        name=arg.run_id,
        config=vars(arg),
    )

if base_model_ckpt in ['None']:
    base_model_ckpt = None
if assistant_model_ckpt in ['None']:
    assistant_model_ckpt = None

model_dtype = torch.bfloat16

base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model_id)

if 'Llama' in base_model_id:
    base_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    base_backbone = 'llama'
elif 'Qwen' in base_model_id:
    base_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    base_backbone = 'qwen'
else:
    raise NotImplementedError
if 'Llama' in assistant_model_id:
    assistant_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    assistant_backbone = 'llama'
elif 'Qwen' in assistant_model_id:
    assistant_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    assistant_backbone = 'qwen'
else:
    raise NotImplementedError

model = EfficientSoftCoTFromSmallModel(
    assistant_model_id,
    base_model_id,
    num_thought_tokens,
    tune_base_model=tune_base_model,
    tune_assistant_model=tune_assistant_model,
    path_to_projection_module=params_file_name,
    path_to_small_language_model=assistant_model_ckpt,
)
logger.info(f'Successfully Init Model `{model.__class__.__name__}`')
model.eval()
model.assistant_model.eval()
model.base_model.eval()

if task_name in ['gsm8k']:
    db = GSM8KLoader().load('data/gsm8k')
    preprocess_method = pre_process_gsm8k
elif task_name in ['strategyqa']:
    db = StrategyQALoader().load()
    preprocess_method = pre_process_strategy_qa
elif task_name in ['asdiv-aug']:
    db = AugASDivLoader().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['aqua']:
    db = AQuALoader().load()
    preprocess_method = pre_process_aqua
elif task_name in ['du']:
    db = DULoader().load()
    preprocess_method = pre_process_du
else:
    raise NotImplementedError

ds = db.get_dataset('test')

if test_k > 0:
    ds = ds[: test_k]

generation_config = GenerationConfig.from_pretrained(base_model_id)
if base_backbone in ['llama']:
    generation_config.pad_token_id = 128009
elif base_backbone in ['qwen']:
    generation_config.pad_token_id = 151643
else:
    raise NotImplementedError
generation_config.top_p = 1.0
generation_config.temperature = 1.0

correct_count = 0
total_trace_tokens = 0
n_logged = 0

for idx, ins in enumerate(tqdm(ds)):

    instance_id = f'{task_name}_test_{idx:05d}'
    if run_logger.is_done(instance_id):
        continue

    t_start = time.time()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    gold_plan_steps = None
    if task_name in ['gsm8k', 'asdiv-aug', 'aqua']:
        raw_answer_field = ins['answer']
        answer = raw_answer_field.split('\n')[-1]
        assert answer.startswith('####')
        gold_plan_steps = len([l for l in raw_answer_field.split('\n')[:-1] if l.strip()])
        answer = answer.replace(',', '')
        if task_name in ['gsm8k', 'asdiv-aug']:
            if '.' in answer:
                answer = float(answer[4:])
            else:
                answer = int(answer[4:])
        else:
            answer = answer[4:].strip()
    elif task_name in ['strategyqa', 'du']:
        answer = ins['answer']
    else:
        raise NotImplementedError

    inputs = preprocess_method(
        ins, base_tokenizer, assistant_tokenizer, num_thought_tokens,
        add_bot_eot=(num_thought_tokens > 0), split='test',
        base_special_token=base_special_token,
        assistant_special_token=assistant_special_token,
        base_backbone=base_backbone,
        assistant_backbone=assistant_backbone,
        device=model.device,
    )

    terminators = [base_tokenizer.eos_token_id]
    if base_backbone in ['llama']:
        terminators.append(base_tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    model_answer_list = []
    model_answer_count = {}
    raw_responses = []

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inputs_embeds = model.base_model.get_input_embeddings()(inputs['input_ids'])

    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    inputs_embeds = model.get_inputs_embeds_for_base_model(
        inputs['assistant_input_ids'],
        inputs['assistant_attention_mask'],
        inputs['input_ids'],
        inputs_embeds,
        inputs['thought_index'],
        print_input,
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    outputs = model.base_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs['attention_mask'],
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        generation_config=generation_config,
        num_return_sequences=num_return_sequences,
    )

    trace_token_counts = []
    for i in range(outputs.shape[0]):
        response = outputs[i]
        trace_token_counts.append(int(response.shape[0]))
        raw_model_answer = base_tokenizer.decode(response, skip_special_tokens=True)
        raw_responses.append(raw_model_answer)

        if task_name in ['gsm8k', 'asdiv-aug']:
            cleaned_model_answer = raw_model_answer.replace(',', '').replace('%', '').replace('$', '')
        else:
            cleaned_model_answer = raw_model_answer

        match = re.findall(r'\s*([\d,]+(?:\.\d+)?)\s*', cleaned_model_answer)

        if task_name in ['gsm8k', 'asdiv-aug']:
            try:
                if match:
                    last_match = match[-1]
                    cleaned_match = last_match.replace(',', '').replace('%', '').replace('$', '')
                    if '.' in cleaned_match:
                        model_answer = round(float(cleaned_match), 2)
                    else:
                        model_answer = int(cleaned_match)
                else:
                    model_answer = None
            except Exception as e:
                model_answer = None
                logger.error(f'Error: {e}')
        elif task_name in ['strategyqa']:
            last_yes = re.search(r'\bsey\b', raw_model_answer.lower()[::-1])
            last_yes = last_yes.start() if last_yes is not None else len(raw_model_answer)
            last_no = re.search(r'\bon\b', raw_model_answer.lower()[::-1])
            last_no = last_no.start() if last_no is not None else len(raw_model_answer)
            model_answer = None if last_yes == last_no == len(raw_model_answer) else (last_yes < last_no)
        elif task_name in ['aqua', 'du']:
            m_answer = re.search(r'\b[a-f]\b', raw_model_answer.lower()[::-1])
            model_answer = m_answer.group(0).upper() if m_answer is not None else None
        else:
            raise NotImplementedError

        model_answer_list.append(model_answer)
        if model_answer in model_answer_count and model_answer is not None:
            model_answer_count[model_answer] += 1
        else:
            model_answer_count[model_answer] = 1

    max_model_count = 0
    final_model_answer = None
    for k, v in model_answer_count.items():
        if v > max_model_count:
            final_model_answer = k
            max_model_count = v

    is_correct = (final_model_answer == answer)
    if is_correct:
        correct_count += 1

    trace_token_count = trace_token_counts[0] if trace_token_counts else None
    reasoning_units = trace_token_count

    run_logger.log({
        'instance_id': instance_id,
        'run_id': arg.run_id,
        'method': 'softcot',
        'task': task_name,
        'backbone': base_model_id,
        'assistant_backbone': assistant_model_id,
        'is_baseline_no_cot': arg.is_baseline_no_cot,
        'seed': seed,
        'gold_answer': answer,
        'gold_plan_steps': gold_plan_steps,
        'model_answer': final_model_answer,
        'correct': bool(is_correct),
        'full_trace': raw_responses[0] if raw_responses else None,
        'trace_token_count': trace_token_count,
        'prefix_token_count': num_thought_tokens,
        'reasoning_units': reasoning_units,
        'latency_s': round(time.time() - t_start, 3),
    })

    n_logged += 1
    total_trace_tokens += (trace_token_count or 0)

    if use_wandb:
        wandb.log({
            "progress": idx + 1,
            "total": len(ds),
            "running_accuracy": correct_count / (idx + 1),
            "trace_token_count": trace_token_count,
            "avg_trace_token_count": total_trace_tokens / max(n_logged, 1),
            "latency_s": round(time.time() - t_start, 3),
        })

    if print_response:
        logger.info(f'({idx + 1}/{len(ds)}) Correct so far: {correct_count}/{idx + 1}')

logger.info(f'Done. Final accuracy: {correct_count}/{len(ds)}')
if use_wandb:
    wandb.log({"final_accuracy": correct_count / len(ds)})
    wandb.finish()
