from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
from data_formatter import get_training_dataset
from utils import get_local_dir
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from typing import Any
import torch.nn.functional as F
import json
import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from typing import Any
import torch.nn.functional as F
import json
import os
from hashlib import md5
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from peft import PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import RobertaModel
import argparse




os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
device_0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device_2 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def generate_batches(data, tokenizer, batch_size, max_length, max_prompt_length):
    collate_fn = get_collate_fn(tokenizer)
    batch = []
    example_idx = 0
    for prompt, responses, pairs, sft_target, rejected_target, truncation_mode in data:
        batch_element = tokenize_batch_element(prompt, sft_target, rejected_target, truncation_mode, tokenizer, max_length, max_prompt_length)
        batch.append(batch_element)
        if len(batch) == batch_size:
            yield collate_fn(batch)
            example_idx += batch_size
            batch = []


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
def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch

def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn
def compute_gradients(model, batch, loss_fn=None,reference_model=None,logit_device = None):
    """
    Compute gradients for the model based on the provided loss function, or use the model's built-in loss.

    Args:
        model (torch.nn.Module): The model for which to compute gradients.
        batch (dict): A dictionary containing the inputs and labels for the model.
        loss_fn (callable, optional): A function that takes model outputs and batch, and returns the loss.
                                      If None, the model's built-in loss will be used.

    Returns:
        torch.Tensor: The vectorized gradients of the model.
    """
    # Ensure the model's parameters are set to require gradients
    # for param in model.parameters():
    #     param.requires_grad = True

    # Forward pass
    # outputs = model(**batch)

    # Calculate loss
    if loss_fn is not None:
        # Use the provided custom loss function
        # loss = loss_fn(outputs, batch)
        loss = calculate_dpo_loss(reference_model=reference_model, policy_model=model, batch = batch,beta =0.1,label_smoothing=0,logit_device= logit_device)
    else:
        # Use the model's built-in loss
        loss = model(**batch).loss

    # Backpropagate to compute gradients
    #model.zero_grad()  # Reset gradients to avoid accumulation
    model.zero_grad()
    loss.backward()

    # Collect and vectorize the gradients
    device = next(model.parameters()).device
    vectorized_grads = torch.cat([
        p.grad.view(-1).to(device) for p in model.parameters() if p.grad is not None
    ])
     
    return vectorized_grads
def load_model(model_name_or_path: str,
               torch_dtype: Any = torch.bfloat16) -> Any:
    """
    Load a model from a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        torch_dtype (Any, optional): The torch data type. Defaults to torch.bfloat16.

    Returns:
        Any: The loaded model.
    """

    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, device_map={"": device_0})
        model = PeftModel.from_pretrained(
            base_model, model_name_or_path, device_map={"": device_0})
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype,device_map={"": device_0})
    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model
def compute_log_probs(model: torch.nn.Module, input_ids: torch.Tensor,attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids, attention_mask=attention_mask).logits
    log_probs = _get_batch_logps(logits, labels, average_log_prob=False)
    return log_probs
def calculate_dpo_loss(reference_model: torch.nn.Module, policy_model: torch.nn.Module, batch: DataLoader,beta =0.1,label_smoothing=0,logit_device: str = "cuda:2") -> float:
    chosen_input_ids = batch['chosen_input_ids'].to(policy_model.device)
    rejected_input_ids = batch['rejected_input_ids'].to(policy_model.device)
    chosen_attention_mask = batch['chosen_attention_mask'].to(policy_model.device)
    rejected_attention_mask = batch['rejected_attention_mask'].to(policy_model.device)
    chosen_labels = batch['chosen_labels'].to(policy_model.device)
    rejected_labels = batch['rejected_labels'].to(policy_model.device)

    policy_chosen_logps = compute_log_probs(policy_model, chosen_input_ids, chosen_attention_mask,chosen_labels).to(logit_device)
    policy_rejected_logps = compute_log_probs(policy_model, rejected_input_ids,rejected_attention_mask,rejected_labels).to(logit_device)

    with torch.no_grad():
        reference_chosen_logps = compute_log_probs(reference_model, chosen_input_ids.to(reference_model.device),chosen_attention_mask.to(reference_model.device),chosen_labels.to(reference_model.device)).to(logit_device)
        reference_rejected_logps = compute_log_probs(reference_model, rejected_input_ids.to(reference_model.device),rejected_attention_mask.to(reference_model.device),rejected_labels.to(reference_model.device)).to(logit_device)
    
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    reference_log_ratios = reference_chosen_logps - reference_rejected_logps
    logits = policy_log_ratios - reference_log_ratios
    #dpo loss
    loss = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    return loss.mean()
def obtain_gradients(model, batch):
    """ obtain gradients. """
    print(model.device)
    for key in batch:
        print(batch[key].device)
    loss = model(**batch).loss
    loss.backward()
    device = next(model.parameters()).device
    vectorized_grads = torch.cat(
        [p.grad.view(-1).to(device) for p in model.parameters() if p.grad is not None])
    return vectorized_grads
def prepare_batch(batch, device= "cpu"):
    """ Move the batch to the device. """
    for key in batch:
        batch[key] = batch[key].to(device)
def prepare_dataset(batch):
    # 这个函数仅保留模型需要的字段
    return {
        'input_ids': batch['input_ids'],
        'labels': batch['labels'],
        'attention_mask': batch['attention_mask']
    }
def _project(current_full_grads, projected_grads):
    current_full_grads = torch.stack(current_full_grads).to(torch.float16)
    for i, projector in enumerate(projectors):
        current_projected_grads = projector.project(
            current_full_grads, model_id=model_id)
        projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

def _save(projected_grads, output_dirs):
    for dim in proj_dim:
        if len(projected_grads[dim]) == 0:
            continue
        projected_grads[dim] = torch.cat(projected_grads[dim])

        output_dir = output_dirs[dim]
        outfile = os.path.join(output_dir, f"grads-{count}.pt")
        torch.save(projected_grads[dim], outfile)
        print(
            f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
        projected_grads[dim] = []
def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector
def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params

def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")

def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")

def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """ 
    Retrieve the highest index for which the data (either representation or gradients) has been stored. 

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """

    files = [file for file in os.listdir(
        output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1])
             for file in files]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


# 定义命令行参数解析器
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--data_path', type=str, required=True, help='Path to the SHP test JSON file')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output gradients')
args = parser.parse_args()

# 读取命令行参数
shp_test_path = args.data_path
model_path = args.model_path
output_dir = args.output_dir


# shp_test_path = "/root/less_research/data/eval/shp/test/shp_preference_test.json"
# 读取 JSON 文件
with open(shp_test_path, 'r') as file:
    data = json.load(file)
reference_model_path = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(reference_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
max_length = 2048 
max_prompt_length = 256
batch_size = 1

flat_data = []
truncation_mode = 'keep_end' if ('hh' in args.data_path) else 'keep_start'
for prompt, values in data.items():
    flat_data.append((prompt, values['responses'], values['pairs'], values['sft_target'], values["rejected_target"],truncation_mode))
collate_fn = get_collate_fn(tokenizer)
test_dataloader =  list(generate_batches(flat_data, tokenizer, batch_size, max_length, max_prompt_length))

print("Data loaded")



# model_path = "/maas-us/notebook/users/yang-2ewu0520/less_result/out/llama2-7b-p0.05-lora-seed3/checkpoint-848"
reference_model = AutoModelForCausalLM.from_pretrained(reference_model_path).to(device_1)
model = load_model(model_path)
model.to(device_0)  # 移动模型到 CUDA 设备



model_id = 0  # model_id is used to draft the random seed for the projectors
block_size = 128  # fixed block size for the projectors
projector_batch_size = 16  # batch size for the projectors
torch.random.manual_seed(0)  # set the random seed for torch

project_interval = 10  # project every 16 batches
save_interval = 10  # save every 160 batches
proj_dim = [8192]



device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
projector = get_trak_projector(device)
number_of_params = get_number_of_params(model)

projectors = []
for dim in proj_dim:
    if projector==BasicProjector:
        proj = projector(grad_dim=number_of_params,
                proj_dim=dim,
                seed=0,
                proj_type=ProjectionType.rademacher,
                device=device,
                dtype=dtype,
                block_size=block_size)
    else:
        proj = projector(grad_dim=number_of_params,
                        proj_dim=dim,
                        seed=0,
                        proj_type=ProjectionType.rademacher,
                        device=device,
                        dtype=dtype,
                        block_size=block_size,
                        max_batch_size=projector_batch_size)
    projectors.append(proj)

count = 0

# output_dir = "/data/less_result/grads/llama2-7b-p0.05-lora-seed3/shp-ckpt848-dpo-sgd"
# set up a output directory for each dimension
output_dirs = {}
for dim in proj_dim:
    output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
    output_dirs[dim] = output_dir_per_dim
    os.makedirs(output_dir_per_dim, exist_ok=True)

# max index for each dimension
max_index = min(get_max_saved_index(
    output_dirs[dim], "grads") for dim in proj_dim)

# projected_gradients
full_grads = []  # full gradients
projected_grads = {dim: [] for dim in proj_dim}  # projected gradients

for batch in tqdm(test_dataloader, total=len(test_dataloader)):
    count += 1
    vectorized_grads= compute_gradients(model, batch, loss_fn="dpo",reference_model=reference_model,logit_device = device_2)
    # add the gradients to the full_grads
    full_grads.append(vectorized_grads)
    if count % project_interval == 0:
        _project(full_grads, projected_grads)
        full_grads = []

    if count % save_interval == 0:
        _save(projected_grads, output_dirs)

if len(full_grads) > 0:
    _project(full_grads, projected_grads)
    full_grads = []

for dim in proj_dim:
    _save(projected_grads, output_dirs)

torch.cuda.empty_cache()
for dim in proj_dim:
    output_dir = output_dirs[dim]
    merge_and_normalize_info(output_dir, prefix="grads")
    merge_info(output_dir, prefix="grads")





#python dpo_gradient.py --data_path /root/less_research/data/eval/shp/test/shp_preference_test.json --model_path /maas-us/notebook/users/yang-2ewu0520/less_result/out/llama2-7b-p0.05-lora-seed3/checkpoint-848 --output_dir /data/less_result/grads/llama2-7b-p0.05-lora-seed3/shp-ckpt848-dpo-sgd