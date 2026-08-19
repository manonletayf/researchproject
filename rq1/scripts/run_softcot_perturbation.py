import sys
import json
import re
import argparse
import time
import torch
from transformers import AutoTokenizer, GenerationConfig

sys.path.append('/home/ubuntu/manon-letayf/SoftCoT')
sys.path.append('/home/ubuntu/manon-letayf/rq1_common')
from llm_model import EfficientSoftCoTFromSmallModel
from data_loader import GSM8KLoader
from utils import pre_process_gsm8k
from logger import RunLogger
from error_generator import corrupt_step

args = argparse.ArgumentParser()
args.add_argument('--source_jsonl', type=str,
                   default='/home/ubuntu/manon-letayf/rq1_results/softcot_gsm8k_seed41.jsonl')
args.add_argument('--base_model_id', type=str, default='Qwen/Qwen2.5-7B-Instruct')
args.add_argument('--assistant_model_id', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
args.add_argument('--params_file_name', type=str,
                   default='checkpoints/SoftCoT/SoftCoT/ckpt-gsm8k-Qwen2.5-7B-Instruct-Qwen2.5-1.5B-Instruct/projection.bin')
args.add_argument('--num_thought_tokens', type=int, default=4)
args.add_argument('--n_instances', type=int, default=20)
args.add_argument('--seed', type=int, default=41)
args.add_argument('--run_id', type=str, required=True)
args.add_argument('--output_jsonl', type=str, required=True)
arg = args.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Chargement des instances correctes deja loggees...')
correct_instances = []
with open(arg.source_jsonl) as f:
    for line in f:
        d = json.loads(line)
        if d.get('correct') and d.get('full_trace'):
            steps = re.findall(r'## Step \d+:.*?(?=## Step \d+:|Therefore|\Z)',
                                d['full_trace'], re.DOTALL)
            if len(steps) >= 2:
                d['steps'] = [s.strip() for s in steps]
                correct_instances.append(d)
        if len(correct_instances) >= arg.n_instances:
            break

print(f'{len(correct_instances)} instances correctes avec >=2 etapes trouvees.')

print('Chargement du dataset GSM8K...')
db = GSM8KLoader().load('data/gsm8k')
ds = db.get_dataset('test')

print('Chargement des tokenizers et du modele...')
base_tokenizer = AutoTokenizer.from_pretrained(arg.base_model_id)
assistant_tokenizer = AutoTokenizer.from_pretrained(arg.assistant_model_id)
base_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
assistant_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']

model = EfficientSoftCoTFromSmallModel(
    arg.assistant_model_id, arg.base_model_id, arg.num_thought_tokens,
    tune_base_model=False, tune_assistant_model=False,
    path_to_projection_module=arg.params_file_name,
    path_to_small_language_model=None,
)
model.eval()
model.assistant_model.eval()
model.base_model.eval()

generation_config = GenerationConfig.from_pretrained(arg.base_model_id)
generation_config.pad_token_id = 151643
generation_config.top_p = 1.0
generation_config.temperature = 1.0

run_logger = RunLogger(arg.output_jsonl)

for d in correct_instances:
    perturb_instance_id = f"perturb_{d['instance_id']}"
    if run_logger.is_done(perturb_instance_id):
        continue

    idx = int(d['instance_id'].split('_')[-1])
    ins = ds[idx]

    corrupt_idx = len(d['steps']) // 2
    original_step = d['steps'][corrupt_idx]
    corrupted_step, corruption_prompt = corrupt_step(original_step)

    prefix_steps = d['steps'][:corrupt_idx]
    reconstructed_text = 'Assistant\nLet\'s solve the problem step-by-step.\n' + \
        '\n'.join(prefix_steps) + '\n' + corrupted_step + '\n'

    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed(arg.seed)

    inputs = pre_process_gsm8k(
        ins, base_tokenizer, assistant_tokenizer, arg.num_thought_tokens,
        add_bot_eot=True, split='test',
        base_special_token=base_special_token,
        assistant_special_token=assistant_special_token,
        base_backbone='qwen', assistant_backbone='qwen',
        device=model.device,
    )

    inputs_embeds = model.base_model.get_input_embeddings()(inputs['input_ids'])
    inputs_embeds = model.get_inputs_embeds_for_base_model(
        inputs['assistant_input_ids'], inputs['assistant_attention_mask'],
        inputs['input_ids'], inputs_embeds, inputs['thought_index'], False,
    )

    corrupted_ids = base_tokenizer.encode(reconstructed_text, return_tensors='pt').to(model.device)
    corrupted_embeds = model.base_model.get_input_embeddings()(corrupted_ids)
    full_embeds = torch.cat([inputs_embeds, corrupted_embeds], dim=1)
    full_attention_mask = torch.ones(full_embeds.shape[:2], device=model.device)

    terminators = [base_tokenizer.eos_token_id]

    with torch.no_grad():
        outputs = model.base_model.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            generation_config=generation_config,
        )

    response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_final_answer(text):
        match = re.findall(r'\\boxed\{([^}]+)\}|\$([0-9.,]+)\$', text)
        if match:
            for m in match[::-1]:
                val = m[0] or m[1]
                if val:
                    return val.replace(',', '').strip()
        return None

    perturbed_answer = extract_final_answer(response)
    gold_answer = d['gold_answer']

    behavioral_sensitivity = (str(perturbed_answer) != str(d.get('model_answer')))
    propagation_visible = False
    corrupted_numbers = re.findall(r'\d+', corrupted_step)
    for num in corrupted_numbers:
        if num in response:
            propagation_visible = True
            break

    run_logger.log({
        'instance_id': perturb_instance_id,
        'run_id': arg.run_id,
        'method': 'softcot',
        'original_instance_id': d['instance_id'],
        'gold_answer': gold_answer,
        'original_model_answer': d.get('model_answer'),
        'corrupt_step_index': corrupt_idx,
        'original_step_text': original_step,
        'corrupted_step_text': corrupted_step,
        'corruption_prompt': corruption_prompt,
        'perturbed_response': response,
        'perturbed_answer': perturbed_answer,
        'behavioral_sensitivity': behavioral_sensitivity,
        'propagation_visible': propagation_visible,
    })
    print(f"{d['instance_id']}: sensitivity={behavioral_sensitivity}, propagation={propagation_visible}")

print('Termine.')
