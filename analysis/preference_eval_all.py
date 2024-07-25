import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict, Union
import json
from preference_dataset import get_batch_iterator
from utils import get_local_dir
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
from tqdm import tqdm
import argparse

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
def compute_log_probs(model: torch.nn.Module, input_ids: torch.Tensor,attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        log_probs = _get_batch_logps(logits, labels, average_log_prob=False)
    return log_probs

def calculate_win_rate(reference_model: torch.nn.Module, policy_model: torch.nn.Module, dataloader: DataLoader,beta =0.1,label_smoothing=0,logit_device: str = "cuda:2") -> float:
    wins = 0
    total = 0
    losses = []
    margins = []
    po_losses = []
    po_margins = []
    po_wins = 0
    for batch in tqdm(dataloader):
        chosen_input_ids = batch['chosen_input_ids'].to(policy_model.device)
        rejected_input_ids = batch['rejected_input_ids'].to(policy_model.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(policy_model.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(policy_model.device)
        chosen_labels = batch['chosen_labels'].to(policy_model.device)
        rejected_labels = batch['rejected_labels'].to(policy_model.device)

        policy_chosen_logps = compute_log_probs(policy_model, chosen_input_ids, chosen_attention_mask,chosen_labels).to(logit_device)
        policy_rejected_logps = compute_log_probs(policy_model, rejected_input_ids,rejected_attention_mask,rejected_labels).to(logit_device)
        reference_chosen_logps = compute_log_probs(reference_model, chosen_input_ids.to(reference_model.device),chosen_attention_mask.to(reference_model.device),chosen_labels.to(reference_model.device)).to(logit_device)
        reference_rejected_logps = compute_log_probs(reference_model, rejected_input_ids.to(reference_model.device),rejected_attention_mask.to(reference_model.device),rejected_labels.to(reference_model.device)).to(logit_device)
        
        policy_log_ratios = policy_chosen_logps - policy_rejected_logps
        reference_log_ratios = reference_chosen_logps - reference_rejected_logps
        logits = policy_log_ratios - reference_log_ratios
        #dpo loss
        loss = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        losses.append(loss)


        #win rate, margin
        chosen_rewards = beta* (policy_chosen_logps-reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
        wins += (chosen_rewards > rejected_rewards).sum().item()
        total += chosen_input_ids.size(0)
        margin = chosen_rewards - rejected_rewards
        margins.append(margin)

        #win rate, margin, dpo loss for policy only
        po_wins += (policy_chosen_logps > policy_rejected_logps).sum().item()
        po_margin = policy_log_ratios * beta
        po_margins.append(po_margin)
        po_loss = -F.logsigmoid(beta * policy_log_ratios) * (1 - label_smoothing) - F.logsigmoid(-beta * policy_log_ratios) * label_smoothing
        po_losses.append(po_loss)






    all_margins = torch.cat(margins)  # 将列表中的所有Tensor连接成一个Tensor
    all_losses = torch.cat(losses)
    win_rate = wins / total

    po_win_rate = po_wins/total
    po_all_margins = torch.cat(po_margins)
    po_all_losses = torch.cat(po_losses)

    return win_rate, wins, total,loss.mean(),all_margins.mean(), po_win_rate, po_all_margins.mean(), po_all_losses.mean()



def main(reference_model_path: str, policy_model_path: str, eval_iterator, max_length: int = 512, batch_size: int = 8):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device_2 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    reference_model = AutoModelForCausalLM.from_pretrained(reference_model_path).to(device_0)
    policy_model = AutoModelForCausalLM.from_pretrained(policy_model_path).to(device_1)

    test_dataloader = list(eval_iterator)
    win_rate, wins, total, loss, margin,po_win_rate, po_margin, po_loss = calculate_win_rate(reference_model, policy_model, test_dataloader,logit_device=device_2)
    print(f"Win Rate: {win_rate:.3f}, Wins count: {wins}, Total count: {total}, Loss: {loss:.3f}, Margin: {margin:.3f}, po win rate: {po_win_rate:.3f}, po margin: {po_margin:.3f}, po loss: {po_loss:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model win rate")
    parser.add_argument("reference_model_path", type=str, help="Path to the reference model")
    parser.add_argument("policy_model_path", type=str, help="Path to the policy model")
    parser.add_argument("--local_file", type=str, default=None,help="Path of local preference dataset")
    args = parser.parse_args()
    # reference_model_path = "/home/ywu19/research/less_research/out/Llama-2-7b-hf"
    # policy_model_path = "/home/ywu19/research/less_research/out/llama2-7b-less-shp_adam_pure-p0.05-lora"
    tokenizer = AutoTokenizer.from_pretrained(args.reference_model_path, cache_dir=get_local_dir(['.cache']))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Assuming eval_iterator is defined elsewhere and passed to main
    eval_iterator = get_batch_iterator(
        names=["shp"],
        tokenizer=tokenizer,
        split='test',
        batch_size=2,
        shuffle=True,
        max_length=2048,
        max_prompt_length=256,
        sft_mode=False,
        n_examples=256,
        seed=42,
        silent=False,
        local = args.local_file,
        cache_dir=get_local_dir(['.cache'])
    )

    main(args.reference_model_path, args.policy_model_path, eval_iterator)



# how to evaluate:
# python preference_eval.py /home/ywu19/research/less_research/out/Llama-2-7b-hf /home/ywu19/out/llama2-7b-p0.05-lora-seed3
