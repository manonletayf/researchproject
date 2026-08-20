import sys
import json
import re
import argparse
from vllm import LLM, SamplingParams
from datasets import load_from_disk

sys.path.append('/home/ubuntu/manon-letayf/rq1_common')
from logger import RunLogger
from error_generator import corrupt_step, OpenAICreditError

args = argparse.ArgumentParser()
args.add_argument('--source_jsonl', type=str,
                   default='/home/ubuntu/manon-letayf/rq1_results/s1_math_batch1.jsonl')
args.add_argument('--dataset_dir', type=str,
                   default='/home/ubuntu/manon-letayf/rq1_results/s1_math_batch1')
args.add_argument('--model_id', type=str, default='simplescaling/s1.1-32B')
args.add_argument('--tokenizer_id', type=str, default='simplescaling/s1-32B')
args.add_argument('--n_instances', type=int, default=20)
args.add_argument('--run_id', type=str, required=True)
args.add_argument('--output_jsonl', type=str, required=True)
arg = args.parse_args()

print('Chargement du dataset source (pour recuperer les questions)...')
source_ds = load_from_disk(arg.dataset_dir)

print('Chargement des instances correctes deja loggees...')
correct_instances = []
with open(arg.source_jsonl) as f:
    for line in f:
        d = json.loads(line)
        if d.get('correct') and d.get('has_thinking_block'):
            trace = d['full_trace']
            if '<|im_start|>think' in trace and '<|im_start|>answer' in trace:
                think_part = trace.split('<|im_start|>think')[-1].split('<|im_start|>answer')[0]
                paragraphs = [p.strip() for p in think_part.split('\n\n') if p.strip()]
                if len(paragraphs) >= 3:
                    d['paragraphs'] = paragraphs
                    correct_instances.append(d)
        if len(correct_instances) >= arg.n_instances:
            break

print(f'{len(correct_instances)} instances correctes avec >=3 paragraphes trouvees.')

print('Chargement du modele vLLM...')
model = LLM(
    model=arg.model_id,
    tokenizer=arg.tokenizer_id,
    tensor_parallel_size=1,
    dtype='bfloat16',
    quantization='bitsandbytes',
    load_format='bitsandbytes',
    gpu_memory_utilization=0.85,
)
tok = model.get_tokenizer()

run_logger = RunLogger(arg.output_jsonl)

for d in correct_instances:
    perturb_instance_id = f"perturb_{d['instance_id']}"
    if run_logger.is_done(perturb_instance_id):
        continue

    corrupt_idx = len(d['paragraphs']) // 2
    original_para = d['paragraphs'][corrupt_idx]
    try:
        corrupted_para, corruption_prompt = corrupt_step(original_para)
    except OpenAICreditError as e:
        print(f'ARRET: {e}')
        print(f'Instances traitees avant arret: voir {arg.output_jsonl}')
        sys.exit(1)

    prefix_paragraphs = d['paragraphs'][:corrupt_idx]
    prefix_text = '\n\n'.join(prefix_paragraphs)

    doc_idx = int(d['instance_id'].split('doc')[-1])
    problem = source_ds[doc_idx]['problem']
    messages = [{'role': 'user', 'content': problem}]
    chat_prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    original_prompt_prefix = chat_prefix + '<|im_start|>think'
    reconstructed = original_prompt_prefix + '\n\n' + \
        '\n\n'.join(prefix_paragraphs) + '\n\n' + corrupted_para

    stop_token_ids = tok('<|im_start|><|im_end|>')['input_ids']
    sampling_params = SamplingParams(
        max_tokens=8192,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    o = model.generate(reconstructed, sampling_params=sampling_params)
    continuation = o[0].outputs[0].text
    continuation_truncated = (len(o[0].outputs[0].token_ids) >= 8192)
    if continuation_truncated:
        print(f"  ATTENTION: continuation tronquee (2048 tokens atteints) pour {d['instance_id']}")

    full_perturbed_prompt = reconstructed + continuation

    answer_stop = tok('<|im_end|>')['input_ids']
    answer_prompt = full_perturbed_prompt + '<|im_start|>answer\n'
    sampling_params_answer = SamplingParams(
        max_tokens=512,
        min_tokens=0,
        stop_token_ids=answer_stop,
        skip_special_tokens=False,
        temperature=0.0,
    )
    o_answer = model.generate(answer_prompt, sampling_params=sampling_params_answer)
    final_answer_text = o_answer[0].outputs[0].text

    def extract_answer(text):
        box = re.findall(r'\\boxed\{([^}]+)\}', text)
        if box:
            return box[-1].strip()
        return None

    perturbed_answer = extract_answer(final_answer_text)
    gold_answer = d['gold_answer']

    behavioral_sensitivity = (str(perturbed_answer) != str(d.get('model_answer')))
    propagation_visible = False
    corrupted_numbers = re.findall(r'\d+', corrupted_para)
    for num in corrupted_numbers:
        if num in continuation:
            propagation_visible = True
            break

    full_perturbed_trace = (prefix_text + '\n\n' if prefix_text else '') + corrupted_para + '\n\n' + continuation

    run_logger.log({
        'instance_id': perturb_instance_id,
        'run_id': arg.run_id,
        'method': 's1',
        'task': 'openai_math',
        'backbone': arg.model_id,
        'original_instance_id': d['instance_id'],
        'gold_answer': gold_answer,
        'original_model_answer': d.get('model_answer'),
        'corrupt_paragraph_index': corrupt_idx,
        'original_paragraph_text': original_para,
        'corrupted_paragraph_text': corrupted_para,
        'corruption_prompt': corruption_prompt,
        'perturbed_continuation': continuation,
        'full_perturbed_trace': full_perturbed_trace,
        'perturbed_final_answer_text': final_answer_text,
        'perturbed_answer': perturbed_answer,
        'behavioral_sensitivity': behavioral_sensitivity,
        'propagation_visible': propagation_visible,
        'continuation_truncated': continuation_truncated,
    })
    print(f"{d['instance_id']}: sensitivity={behavioral_sensitivity}, propagation={propagation_visible}")

print('Termine.')
