import subprocess
import numpy as np
import csv
import os
import argparse
from tqdm import tqdm
import pandas as pd

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def get_validation_loss(model_path, reference_model_path, task):
    command = f"python get_val_loss_all.py --model_path {model_path} --reference_model_path {reference_model_path} --target_task {task}"
    output = run_command(command)
    val_loss = float(output.split(": ")[-1].strip())
    return val_loss

def preference_eval(model_path, ckpt_path, local= None,local_file_path=None):
    if local is None:
        command = f"python preference_eval_all.py {model_path} {ckpt_path}"
    else:
        command = f"python preference_eval_all.py {model_path} {ckpt_path} --local_file {local_file_path}"
    output = run_command(command)
    metrics = {}
    for line in output.split(","):
        key, value = line.strip().split(":")
        metrics[key.strip()] = float(value.strip())
    return metrics


def calculate_statistics(results):
    stats = {}
    for key in results[0].keys():
        values = [result[key] for result in results]
        mean = np.mean(values)
        std = np.std(values)
        stats[key] = (mean, std)
    return stats

def save_results_to_csv(ckpt_path, val_losses, eval_results, stats, task_name):
    csv_path = os.path.join(ckpt_path, f"{task_name}evaluation_results.csv")

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Run1", "Run2", "Run3", "Run4", "Run5", "Mean", "Std"])
        
        for key in eval_results[0].keys():
            row = [key]
            for result in eval_results:
                row.append(result[key])
            row.append(stats[key][0])
            row.append(stats[key][1])
            writer.writerow(row)
        
        writer.writerow(["Validation Loss"] + val_losses + [np.mean(val_losses), np.std(val_losses)])
    data = pd.read_csv(csv_path)
    data.iloc[0,0] = "Win rate"
    data.iloc[3,0] = "DPO loss"
    data.to_csv(csv_path,index = False)
    print(f"Evaluation results saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--method", type=str, default = "less", help="less or less_dpo")
    parser.add_argument("--target_task", type=str, default = "shp", help="target task to evaluate, can be shp, hh, se")
    parser.add_argument("--base_model_path", type=str, default = "meta-llama/Llama-2-7b", help="base model path")
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    task = args.target_task
    model_path = args.base_model_path


    if task=="shp":
        local_file_path = "../data/eval/shp_all/dev/shp_all_5_shot_preference_val.json"
    elif task=="hh":
        local_file_path = "../data/eval/hh_5_shot/dev/hh_preference_val.json"
    elif task=="se":
        local_file_path = "../data/eval/se_5_shot/dev/se_preference_val.json"
    
    val_losses = []
    eval_results = []

    for _ in tqdm(range(5)):
        if args.method=="less":
            val_loss = get_validation_loss(ckpt_path, model_path, task)
            val_losses.append(val_loss)
        else:
            val_loss = preference_eval(model_path, ckpt_path, local= "1", local_file_path=local_file_path)["Loss"]
            val_losses.append(val_loss)
        result = preference_eval(model_path, ckpt_path)
        eval_results.append(result)

    stats = calculate_statistics(eval_results)
    save_results_to_csv(ckpt_path, val_losses, eval_results, stats, task)

if __name__ == "__main__":
    main()
