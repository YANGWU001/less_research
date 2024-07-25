import csv
import json
import random
import argparse
import os
from tqdm import tqdm

def parse_args():
    argparser = argparse.ArgumentParser(description='Script for sampling and creating a JSONL file')
    argparser.add_argument('--sorted_csv', type=str, required=True, help='Path to the sorted CSV file')
    argparser.add_argument('--train_files', type=str, nargs='+', required=True, help='List of paths to the training files')
    argparser.add_argument('--output_jsonl', type=str, required=True, help='Path to the output JSONL file')
    argparser.add_argument('--num_samples', type=int, default=None, help='Number of samples to select')
    args = argparser.parse_args()
    return args

def read_sorted_csv(csv_path):
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            samples.append(row)
    return samples

def random_sample(samples, num_samples):
    return random.sample(samples, num_samples)

def read_training_files(train_files):
    train_data = {}
    for file_path in train_files.values():
        with open(file_path, 'r', encoding='utf-8') as file:
            train_data[os.path.basename(file_path)] = file.readlines()
    return train_data

def write_jsonl(samples, train_data, output_jsonl):
    data = []
    for sample in tqdm(samples):
        file_name = sample['file name']
        index = int(sample[' index'])
        lines = train_data[file_name+"_data.jsonl"]
        data.append(json.loads(lines[index]))

    with open(output_jsonl, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    args = parse_args()

    # Read sorted CSV
    samples = read_sorted_csv(args.sorted_csv)

    # Randomly sample 20 samples
    # sampled = random_sample(samples, args.num_samples)
    if args.num_samples is not None:
        sampled = random_sample(samples, args.num_samples)
    else:
        sampled = random_sample(samples, int(len(samples)*0.05))

    # Map file names to training file paths
    train_files_dict = {os.path.basename(f): f for f in args.train_files}
    # Read training files once
    train_data = read_training_files(train_files_dict)

    # Write the sampled data to a JSONL file
    write_jsonl(sampled, train_data, args.output_jsonl)

    # Write the sampled data to a JSONL file
    #write_jsonl(sampled, train_files_dict, args.output_jsonl)
