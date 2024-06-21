# Less_research

## Step 1. warmup training
```setup
data_dir=../data
model_path=meta-llama/Llama-2-7b-hf
percentage=0.0001 # percentage of the full data to train, you can specify the training file you want to use in the script
data_seed=3
job_name=llama2-7b-p${percentage}-lora-seed${data_seed}

./step1_warmup_lora.sh "$data_dir" "$model_path" "$percentage" "$data_seed" "$job_name"
```

## Step 2. training storage
```bash
ckpt=3
training_data_name=dolly
training_data_file=../data/train/processed/dolly/dolly_data.jsonl
gradient_type="adam"
model_path=../out/llama2-7b-p0.0001-lora-seed3/checkpoint-${ckpt}
output_path=../grads/llama2-7b-p0.0001-lora-seed3/${training_data_name}-ckpt${ckpt}-${gradient_type}
dims="8192"
max_samples=100 #if set to None, run all samples

./step2_training_storage.sh "$training_data_file" "$model_path" "$output_path" "$dims" "$gradient_type" "$max_samples"
```

## Step 3. validate storage
```bash
ckpt=3
task=mmlu
model_path=../out/llama2-7b-p0.0001-lora-seed3/checkpoint-${ckpt}
output_path=../grads/llama2-7b-p0.0001-lora-seed3/${task}-ckpt${ckpt}-sgd # for validation data, we always use sgd
data_dir=../data
dims="8192" # We use 8192 as our default projection dimension 

./step3_validate_storage.sh "$task" "$data_dir" "$model_path" $output_path "$dims"
```

## Step 4. influence calculation
```bash
dim=8192 # decide which dimension to use
gradient_path=../grads/llama2-7b-p0.0001-lora-seed3/{}-ckpt{}-adam/dim${dim}
train_file_names="dolly" #可以是多个train 的file，用空格隔开
#ckpts="105 211 317 420" # checkpoing index，可以选好几个check point
ckpts="3"
checkpoint_weights="1.0e-05" # average lr of the epoch，相应的lr

validation_gradient_path=../grads/llama2-7b-p0.0001-lora-seed3/{}-ckpt{}-sgd/dim${dim}
target_task_names="mmlu"
selected_data_output_path="../selected_data"

./step4_influence_calculation.sh "$gradient_path" "$train_file_names" "$ckpts" "$checkpoint_weights" "$validation_gradient_path" "$target_task_names" "$selected_data_output_path"
```

## Step 5. save selected data
```bash
target_task_name="mmlu"
train_file_names="dolly" #可以是多个train 的file，用空格隔开
train_files="../data/train/processed/dolly/dolly_data.jsonl"
output_path="../selected_data"
percentage=0.05

./step5_save_selected_data.sh "$target_task_name" "$train_file_names" "$train_files" "$output_path" "$percentage"
```

## Step 6. finetune
```bash
target_task_name="mmlu"
percentage=0.05
train_files=../selected_data/${target_task_name}/top_p${percentage}.jsonl
model_path=meta-llama/Llama-2-7b-hf
job_name=llama2-7b-less-mmlu-p${PERCENTAGE}-lora

./step6_finetune.sh "$train_files" "$model_path" "$job_name" 
```

## Step 7. evaluate
```bash
source eval_bbh.sh 
DATA_DIR=/home/ywu19/less_data/data/eval
MDIR=/home/ywu19/out/llama2-7b-less-p0.05-lora
eval_bbh "$MDIR" "$DATA_DIR"

#结果会保存在 MDIR/eval/task name/metrics.json 当中，也可以看log.txt
```
