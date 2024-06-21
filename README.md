# Less_research

## Step 1. warmup training
```setup
data_dir=../less_data/data
model_path=meta-llama/Llama-2-7b-hf
percentage=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
data_seed=3
job_name=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}

./less/scripts/train/warmup_lora_train.sh "$data_dir" "$model_path" "$percentage" "$data_seed" "$job_name"
```

## Step 2. training storage
```bash
CKPT=1688
TRAINING_DATA_NAME=dolly
TRAINING_DATA_FILE=../less_data/data/train/processed/dolly/dolly_data.jsonl
GRADIENT_TYPE="adam"
MODEL_PATH=../out/llama2-7b-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/llama2-7b-p0.05-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
```

## Step 3. validate storage
```bash
CKPT=1688
TASK=tydiqa
MODEL_PATH=../out/llama2-7b-p0.05-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/llama2-7b-p0.05-lora-seed3/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
DATA_DIR=../less_data/data
DIMS="8192" # We use 8192 as our default projection dimension 

./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"
```

## Step 4. influence calculation
```bash
DIM=8192 # decide which dimension to use
GRADIENT_PATH=../grads/llama2-7b-p0.05-lora-seed3/{}-ckpt{}-adam/dim${DIM}
TRAIN_FILE_NAMES="dolly cot flan_v2 oasst1" #可以是多个train 的file，用空格隔开
#CKPTS="105 211 317 420" # checkpoing index，可以选好几个check point
CKPTS="1688"
CHECKPOINT_WEIGHTS="1.0e-05" # average lr of the epoch，相应的lr

VALIDATION_GRADIENT_PATH=../grads/llama2-7b-p0.05-lora-seed3/{}-ckpt{}-sgd/dim${DIM}
TARGET_TASK_NAMES="tydiqa"
SELECTED_DATA_OUTPUT_PATH="../selected_data"

./less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"
```

## Step 5. save selected data
```bash
TRAIN_FILE_NAMES="dolly cot flan_v2 oasst1"
python3 -m less.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files ../less_data/data/train/processed/dolly/dolly_data.jsonl ../less_data/data/train/processed/cot/cot_data.jsonl ../less_data/data/train/processed/flan_v2/flan_v2_data.jsonl ../less_data/data/train/processed/oasst1/oasst1_data.jsonl \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage 0.05
```

## Step 6. finetune
```bash
TARGET_TASK_NAME="tydiqa"
PERCENTAGE=0.05
TRAIN_FILES=../selected_data/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-7b-hf
JOB_NAME=llama2-7b-less-tydiqa-p${PERCENTAGE}-lora

./less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 
```

## Step 7. evaluate
```bash
source eval_bbh.sh 
DATA_DIR=/home/ywu19/less_data/data/eval
MDIR=/home/ywu19/out/llama2-7b-less-p0.05-lora
eval_bbh "$MDIR" "$DATA_DIR"

#结果会保存在 MDIR/eval/task name/metrics.json 当中，也可以看log.txt
```
