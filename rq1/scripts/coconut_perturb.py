import torch
import sys
sys.path.append('/home/ubuntu/manon-letayf/coconut')
from coconut import Coconut, Outputs, MAX_N_LATENT
from torch.nn import CrossEntropyLoss


class CoconutPerturb(Coconut):
    def forward(self, input_ids, attention_mask, labels, position_ids,
                perturb_pass_idx=None, perturb_mode='noise', perturb_scale=1.0,
                **kwargs):
        logits = []
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists])
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)
            next_compute_range = (
                next_compute_range[1],
                input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1,
            )
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                latent_vec = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]

                if perturb_pass_idx is not None and pass_idx == perturb_pass_idx:
                    if perturb_mode == 'noise':
                        noise = torch.randn_like(latent_vec) * latent_vec.norm() * perturb_scale
                        latent_vec = latent_vec + noise
                    elif perturb_mode == 'zero':
                        latent_vec = torch.zeros_like(latent_vec)
                    else:
                        raise ValueError(f'perturb_mode inconnu: {perturb_mode}')

                tensor_list[batch_idx][token_idx] = latent_vec

            inputs_embeds = torch.stack([
                torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])
            ])

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=(
                [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                if kv_cache else None
            ),
            output_hidden_states=True,
        )
        logits.append(outputs.logits)
        self.gen_forward_cnt += max_n_latents + 1
        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def generate_perturbed(self, input_ids, attention_mask, max_new_tokens=64,
                            perturb_pass_idx=None, perturb_mode='noise', perturb_scale=1.0):
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "batch_size==1 uniquement"
        tokens = input_ids[0].detach().tolist()
        labels = input_ids.clone()

        outputs = self.forward(
            input_ids, torch.ones_like(input_ids, device=input_ids.device), labels,
            torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).reshape(1, -1),
            perturb_pass_idx=perturb_pass_idx, perturb_mode=perturb_mode, perturb_scale=perturb_scale,
        )
        inputs_embeds = outputs.inputs_embeds
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        return torch.tensor(tokens).view(1, -1)
