#!/bin/bash


# 重定向所有输出到日志文件
exec > >(tee -i experiment_all_ckpt_log.txt)
exec 2>&1

data_seed=$1
percentage=$2
model_load_path=$3
devices=$4
max_collect_samples=$5
projection_dims=$6


model_name=$(basename "$model_load_path")

# step 1: warmup
function run_step1 {
    data_dir=../data
    model_path=$model_load_path
    data_seed=$1
    job_name=${model_name}-p${percentage}-warmup-lora-seed${data_seed}
    device=devices
    echo "Running step 1 with seed $data_seed"
    ./step1_warmup_lora.sh "$data_dir" "$model_path" "$percentage" "$data_seed" "$job_name" "$device"
    if [ $? -ne 0 ]; then
        echo "Step 1 failed for seed $data_seed" >> experiment_errors.log
        exit 1
    fi
}

function get_checkpoints {
    model_dir=$1
    checkpoints=($(ls "$model_dir" | grep "checkpoint-" | awk -F '-' '{print $2}' | sort -n))
    echo "${checkpoints[@]}"
}

# step 2: training storage
function run_step2 {
    export CUDA_VISIBLE_DEVICES=$1
    ckpt=$2
    training_data_name=$3
    training_data_file=$4
    gradient_type=$5
    model_path=../out/${model_name}-p${percentage}-warmup-lora-seed${data_seed}/checkpoint-${ckpt}
    output_path=../grad/${model_name}-p${percentage}-warmup-lora-seed${data_seed}/${training_data_name}-ckpt${ckpt}-${gradient_type}
    dims=projection_dims
    max_samples=max_collect_samples

    echo "Running step 2 with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, checkpoint=$ckpt, data=$training_data_name, gradient=$gradient_type"
    ./step2_training_storage.sh "$training_data_file" "$model_path" "$output_path" "$dims" "$gradient_type" "$max_samples"
    if [ $? -ne 0 ]; then
        echo "Step 2 failed for checkpoint $ckpt, data $training_data_name, gradient $gradient_type" >> experiment_errors.log
        exit 1
    fi
    echo "Step 2 completed for checkpoint $ckpt, data $training_data_name, gradient $gradient_type"
}

# step 3: less validation storage
function run_step3 {
    export CUDA_VISIBLE_DEVICES=$1
    ckpt=$2
    training_data_name=$3
    training_data_file=$4
    gradient_type=$5
    model_path=../out/${model_name}-p${percentage}-warmup-lora-seed${data_seed}/checkpoint-${ckpt}
    output_path=../grad/${model_name}-p${percentage}-warmup-lora-seed${data_seed}/${training_data_name}-ckpt${ckpt}-${gradient_type}
    dims=projection_dims
    max_samples=max_collect_samples

    echo "Running step 3 with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, checkpoint=$ckpt, data=$training_data_name, gradient=$gradient_type"
    ./step2_training_storage.sh "$training_data_file" "$model_path" "$output_path" "$dims" "$gradient_type" "$max_samples"
    if [ $? -ne 0 ]; then
        echo "Step 3 failed for checkpoint $ckpt, data $training_data_name, gradient $gradient_type" >> experiment_errors.log
        exit 1
    fi
}

# step 3: dpo validation storage
function run_step3_dpo {
    export CUDA_VISIBLE_DEVICES=$1
    ckpt=$2
    validation_data_name=$3
    data_path=$4
    gradient_type=$5
    
    model_path=../out/${model_name}-p${percentage}-warmup-lora-seed${data_seed}/checkpoint-${ckpt}
    output_path=../grad/${model_name}-p${percentage}-warmup-lora-seed${data_seed}/${validation_data_name}-ckpt${ckpt}-dpo-${gradient_type}

    echo "Running step 3 DPO with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, checkpoint=$ckpt, data=$validation_data_name, gradient=$gradient_type"
    python ../analysis/dpo_gradient.py \
        --data_path "$data_path" \
        --model_path "$model_path" \
        --output_dir "$output_path"
    if [ $? -ne 0 ]; then
        echo "Step 3 DPO failed for checkpoint $ckpt, data $validation_data_name, gradient $gradient_type" >> experiment_errors.log
        exit 1
    fi
}

# step 4: influence calculation
function run_step4 {
    export CUDA_VISIBLE_DEVICES=$1
    ckpt=$2
    validation_data_name=$3
    gradient_type=$4
    checkpoint_weights=$5
    dim=projection_dims
    gradient_path=../grad/${model_name}-p${percentage}-lora-seed${data_seed}/{}-ckpt{}-${gradient_type}/dim${dim}
    validation_gradient_path=../grad/${model_name}-p${percentage}-lora-seed${data_seed}/{}-ckpt{}-sgd/dim${dim}
    target_task_names=$6
    selected_data_output_path="../less_result/less_data"

    echo "Running step 4 with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, checkpoint=$ckpt, data=$validation_data_name, gradient=$gradient_type"
    ./step4_influence_calculation.sh "$gradient_path" "$validation_data_name" "$ckpt" "$checkpoint_weights" "$validation_gradient_path" "$target_task_names" "$selected_data_output_path"
    if [ $? -ne 0 ]; then
        echo "Step 4 failed for checkpoint $ckpt, data $validation_data_name, gradient $gradient_type" >> experiment_errors.log
        exit 1
    fi
}

# step 4: dpo influence calculation
function run_step4_dpo {
    export CUDA_VISIBLE_DEVICES=$1
    ckpt=$2
    training_data_name=$3
    gradient_type=$4
    checkpoint_weights=$5
    dim=projection_dims
    gradient_path=../grad/${model_name}-p${percentage}-lora-seed${data_seed}/{}-ckpt{}-${gradient_type}/dim${dim}
    validation_gradient_path=../grad/${model_name}-p${percentage}-lora-seed${data_seed}/{}-ckpt{}-dpo-sgd/dim${dim}
    target_task_names=$6
    selected_data_output_path="../less_result/less_dpo_data"

    echo "Running step 4 DPO with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, checkpoint=$ckpt, data=$training_data_name, gradient=$gradient_type"
    ./step4_influence_calculation.sh "$gradient_path" "$training_data_name" "$ckpt" "$checkpoint_weights" "$validation_gradient_path" "$target_task_names" "$selected_data_output_path"
    if [ $? -ne 0 ]; then
        echo "Step 4 DPO failed for checkpoint $ckpt, data $training_data_name, gradient $gradient_type" >> experiment_errors.log
        exit 1
    fi
}

# step 5: save selected data
function run_step5 {
    export CUDA_VISIBLE_DEVICES=$1
    ckpt=$2
    target_task_name=$3
    train_file_names="dolly cot flan_v2 oasst1"
    train_files="../data/train/processed/dolly/dolly_data.jsonl ../data/train/processed/cot/cot_data.jsonl ../data/train/processed/flan_v2/flan_v2_data.jsonl ../data/train/processed/oasst1/oasst1_data.jsonl"
    output_path="../less_result/less_data"
    echo "Running step 5 with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    ./step5_save_selected_data.sh "$target_task_name" "$train_file_names" "$train_files" "$output_path" "${percentage}"
    if [ $? -ne 0 ]; then
        echo "Step 5 failed" >> experiment_errors.log
        exit 1
    fi
}

# step 5: save selected data for dpo
function run_step5_dpo {
    export CUDA_VISIBLE_DEVICES=$1
    ckpt=$2
    target_task_name=$3
    train_file_names="dolly cot flan_v2 oasst1"
    train_files="../data/train/processed/dolly/dolly_data.jsonl ../data/train/processed/cot/cot_data.jsonl ../data/train/processed/flan_v2/flan_v2_data.jsonl ../data/train/processed/oasst1/oasst1_data.jsonl"
    output_path="../less_result/less_dpo_data"
    echo "Running step 5 DPO with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    ./step5_save_selected_data.sh "$target_task_name" "$train_file_names" "$train_files" "$output_path" "${percentage}"
    if [ $? -ne 0 ]; then
        echo "Step 5 DPO failed" >> experiment_errors.log
        exit 1
    fi
}

# step 6: finetune
function run_step6 {
    train_files=$1
    model_path=$model_load_path
    job_name=$2
    device=devices

    echo "Running step 6 with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, job_name=$job_name"
    ./step6_finetune.sh "$train_files" "$model_path" "$job_name" "$device"
    if [ $? -ne 0 ]; then
        echo "Step 6 failed for job $job_name" >> experiment_errors.log
        exit 1
    fi
}



# step 7: evaluate
function run_step7 {
    ckpt_path=$1
    eval_task=$2
    cd ../analysis
    python evaluate_model_all.py \
        --ckpt_path "$ckpt_path" \
        --target_task "$eval_task"\
        --base_model_path "$model_load_path"
    if [ $? -ne 0 ]; then
        echo "Step 7 failed for checkpoint $ckpt_path" >> experiment_errors.log
        exit 1
    fi
    cd -
}

function run_dpo_step7 {
    ckpt_path=$1
    eval_task=$2
    cd ../analysis
    python evaluate_model_all.py \
        --ckpt_path "$ckpt_path" \
        --method dpo \
        --target_task "$eval_task" \
        --base_model_path "$model_load_path"
    if [ $? -ne 0 ]; then
        echo "Step 7 failed for checkpoint $ckpt_path" >> experiment_errors.log
        exit 1
    fi
    cd -
}









# Begin training

#step 1 warm up lora
run_step1 $data_seed


model_dir=../out/${model_name}-p${percentage}-warmup-lora-seed${data_seed}
checkpoints=($(get_checkpoints "$model_dir" | sort -n))
checkpoints_str=$(IFS=' '; echo "${checkpoints[*]}")
echo "Checkpoints for seed $data_seed: $checkpoints_str"


file_path="${model_dir}/trainer_state.json"

# 运行 Python 脚本并获取输出
learning_rates=$(python3 ../analysis/get_learning_rates.py "$file_path")

# 将输出转换为数组
learning_rates_array=($learning_rates)
learning_rates_str=$(IFS=' '; echo "${learning_rates_array[*]}")
echo "Checkpoints' weights for seed $data_seed: $learning_rates_str"

# step 2 save training gradient
for ckpt in "${checkpoints[@]}"; do
    echo "Starting step 2 for checkpoint $ckpt"
    run_step2 0 $ckpt "dolly" "../data/train/processed/dolly/dolly_data.jsonl" "adam" &
    pid1=$!
    run_step2 1 $ckpt "dolly" "../data/train/processed/dolly/dolly_data.jsonl" "sgd" &
    pid2=$!
    run_step2 2 $ckpt "cot" "../data/train/processed/cot/cot_data.jsonl" "adam" &
    pid3=$!
    run_step2 3 $ckpt "cot" "../data/train/processed/cot/cot_data.jsonl" "sgd" &
    pid4=$!
    run_step2 4 $ckpt "flan_v2" "../data/train/processed/flan_v2/flan_v2_data.jsonl" "adam" &
    pid5=$!
    run_step2 5 $ckpt "flan_v2" "../data/train/processed/flan_v2/flan_v2_data.jsonl" "sgd" &
    pid6=$!
    run_step2 6 $ckpt "oasst1" "../data/train/processed/oasst1/oasst1_data.jsonl" "adam" &
    pid7=$!
    run_step2 7 $ckpt "oasst1" "../data/train/processed/oasst1/oasst1_data.jsonl" "sgd" &
    pid8=$!
    
    wait $pid1
    wait $pid2
    wait $pid3
    wait $pid4
    wait $pid5
    wait $pid6
    wait $pid7
    wait $pid8
    echo "Completed step 2 for checkpoint $ckpt"
done

# step 3 save evaluate gradient
for ckpt in "${checkpoints[@]}"; do
    run_step3 0 $ckpt "shp" "../data/eval/shp_all/dev/shp_all_5_shot_sft_val.json" "sgd" &
    pid1=$!
    run_step3_dpo 1 $ckpt "shp" "../data/eval/shp_all/dev/shp_all_5_shot_preference_val.json" "sgd" &
    pid2=$!
    run_step3 2 $ckpt "hh" "../data/eval/hh_5_shot/dev/hh_sft_val.json" "sgd" &
    pid3=$!
    run_step3_dpo 3 $ckpt "hh" "../data/eval/hh_5_shot/dev/hh_preference_val.json" "sgd" &
    pid4=$!
    run_step3 4 $ckpt "se" "../data/eval/se_5_shot/dev/se_sft_val.json" "sgd" &
    pid5=$!
    run_step3_dpo 5 $ckpt "se" "../data/eval/se_5_shot/dev/se_preference_val.json" "sgd" &
    pid6=$!
    wait $pid1
    wait $pid2
    wait $pid3
    wait $pid4
    wait $pid5
    wait $pid6
    echo "Completed step 3 for checkpoint $ckpt"
done


# step 4 calculate influence score

run_step4 0 "$checkpoints_str" "dolly cot flan_v2 oasst1"  "adam" "$learning_rates_str" "shp" &
pid1=$!
run_step4_dpo 1 "$checkpoints_str" "dolly cot flan_v2 oasst1"  "adam" "$learning_rates_str" "shp" &
pid2=$!
run_step4 2 "$checkpoints_str" "dolly cot flan_v2 oasst1"  "adam" "$learning_rates_str" "hh" &
pid3=$!
run_step4_dpo 3 "$checkpoints_str" "dolly cot flan_v2 oasst1"  "adam" "$learning_rates_str" "hh" &
pid4=$!
run_step4 4 "$checkpoints_str" "dolly cot flan_v2 oasst1"  "adam" "$learning_rates_str" "se" &
pid5=$!
run_step4_dpo 5 "$checkpoints_str" "dolly cot flan_v2 oasst1"  "adam" "$learning_rates_str" "se" &
pid6=$!
wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5
wait $pid6
echo "Completed step 4 for checkpoint $ckpt"


# step 5 save selected data
ckpt=-all
run_step5 0 $ckpt "shp" &
pid1=$!
run_step5_dpo 1 $ckpt "shp" &
pid2=$!
run_step5 2 $ckpt "hh" &
pid3=$!
run_step5_dpo 3 $ckpt "hh" &
pid4=$!
run_step5 4 $ckpt "se" &
pid5=$!
run_step5_dpo 5 $ckpt "se" &
pid6=$!
wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5
wait $pid6
echo "Completed step 5 for seed $seed"




targets=("shp" "hh" "se")
ckpt=-all
for target_name in "${targets[@]}"; do
    # step 6 finetune for less
    less_train_files="../less_result/less_data/${target_name}/top_p${percentage}.jsonl"
    less_job_name="${model_name}-less-${target_name}-p${percentage}-lora-seed${data_seed}-ckpt${ckpt}"
    run_step6 $less_train_files $less_job_name
    echo "Completed step 6 for less model seed $seed"

    #step7 evaluate less model
    less_ckpt_path="../out/${model_name}-less-${target_name}-p${percentage}-lora-seed${data_seed}-ckpt${ckpt}"
    run_step7 $less_ckpt_path $target_name
    echo "Completed step 7 for less model seed $seed"


    # step 6 finetune for less_dpo
    dpo_train_files="../less_result/less_dpo_data/${target_name}/top_p${percentage}.jsonl"
    dpo_job_name="${model_name}-less-dpo-${target_name}-p${percentage}-lora-seed${data_seed}-ckpt${ckpt}"
    run_step6 $dpo_train_files $dpo_job_name
    echo "Completed step 6 for less dpo model seed $seed"

    #step7 evaluate less dpo model
    dpo_ckpt_path="../out/${model_name}-less-dpo-${target_name}-p${percentage}-lora-seed${data_seed}-ckpt${ckpt}"
    run_dpo_step7 $dpo_ckpt_path $target_name
    echo "Completed step 7 for less dpo model seed $seed"
    

done




