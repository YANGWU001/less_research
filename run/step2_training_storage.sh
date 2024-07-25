#!/bin/bash

train_file=$1 #
model=$2 # path to model
output_path=$3 # path to output
dims=$4 # dimension of projection, can be a list
gradient_type=$5
max_samples=$6

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

command="python ../less/build_storage.py \
--train_file $train_file \
--info_type grads \
--model_path $model \
--output_path $output_path \
--gradient_projection_dimension $dims \
--gradient_type $gradient_type"

# Add max_samples argument only if it's not None
if [[ $max_samples != "None" ]]; then
    command+=" --max_samples $max_samples"
fi

eval $command
