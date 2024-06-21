#!/bin/bash
target_task_name=$1
train_file_names=$2
train_files=$3
output_path=$4
percentage=$5

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi


python3 ../less/data_selection_save.py \
--target_task_names $target_task_name \
--train_file_names $train_file_names \
--train_files $train_files \
--output_path $output_path \
--percentage $percentage