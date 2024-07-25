from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch
import argparse
from tqdm import tqdm
import os
from data_formatter import get_training_dataset
from utils import get_local_dir

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
parser = argparse.ArgumentParser(description="Evaluate the model on validation data")

parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
parser.add_argument("--reference_model_path", type=str, required=True, help="Path to the reference model")
parser.add_argument("--target_task", type=str, default = "shp",required=True, help="target task, can be shp, hh, se")

args = parser.parse_args()

# 路径和配置
reference_model_path = args.reference_model_path
# model_path = "/home/ywu19/research/less_research/out/llama2-7b-less-shp_adam_pure-p0.05-lora"

target = args.target_task

if target=="shp":
    data_file_path = '../data/eval/shp_all/dev/shp_all_5_shot_sft_val.json'
elif target=="hh":
    data_file_path = '../data/eval/hh_5_shot/dev/hh_sft_val.json'
elif target=="se":
    data_file_path = '../data/eval/se_5_shot/dev/se_sft_val.json'

# 初始化 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(reference_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(args.model_path)
model.to(device)  # 移动模型到 CUDA 设备
model.eval()

# 加载验证数据集

val_dataset = get_training_dataset([data_file_path],
                                   tokenizer=tokenizer,
                                   max_seq_length=2048,
                                   sample_percentage=1.0,
                                   seed=10)
def prepare_dataset(batch):
    # 这个函数仅保留模型需要的字段
    return {
        'input_ids': batch['input_ids'],
        'labels': batch['labels'],
        'attention_mask': batch['attention_mask']
    }

# 加载数据集，这里假设你已经有了val_dataset
# val_dataset = load_dataset(...) # 如果需要加载数据集的代码

# 使用 map 函数预处理数据集
val_dataset = val_dataset.map(prepare_dataset, batched=True, remove_columns=val_dataset.column_names)
# 使用 DataCollator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest", model=model)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=data_collator)

# 计算验证集上的平均损失
total_loss = 0
total_samples = 0

for batch in tqdm(val_dataloader):
    # 移动数据到 CUDA 设备
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        loss = outputs.loss

    total_loss += loss.item() * batch['input_ids'].size(0)
    total_samples += batch['input_ids'].size(0)

average_loss = total_loss / total_samples
print(f"The validation loss of the model is: {average_loss:.3f}")


#python get_val_loss.py --model_path /home/ywu19/research/less_research/out/llama2-7b-less-shp_adam_pure-p0.05-lora