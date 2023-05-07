#!/bin/bash

selected_model_type="camelyon16"

selected_model_configs="conf/selected_model/$selected_model_type"

device=$1
model=$2

# Run for 5 seeds
for seed in {0..4}; do
    echo "Running $model with seed $seed"
    /mil_env/bin/python train.py +selected_model/$selected_model_type=$model seed=$seed device=$device
done