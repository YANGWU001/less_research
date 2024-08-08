#!/bin/bash


# 重定向所有输出到日志文件



data_seed=666  # Experiment seed
percentage=0.05   # Data percentage to select
model_load_path=meta-llama/Llama-2-7b-hf    # Your base LLM architechture here
devices="0 1 2 3" # Cuda devices available to use
max_collect_samples=None  # The number of training data you want to test the code, after everything works, you can set it to None to run on all training data
projection_dims=8192  # The projection dimension

./run_less_and_less_dpo.sh "$data_seed" "$percentage" "$model_load_path" "$devices" "$max_collect_samples" "$projection_dims"
