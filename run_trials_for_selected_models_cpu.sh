#!/bin/bash

selected_model_type="mnist_collage"
# selected_model_type="mnist_collage_ablations"

selected_model_configs="conf/selected_model/$selected_model_type"

dataset="mnist_collage"
dataset="mnist_collage_inverse"

group_prefix="selected"

# Iterate over yaml files
for yaml_file in $selected_model_configs/*.yaml; do
    # yaml_file=$selected_model_configs/distance_aware_self_attention.yaml
    # Extract model name
    model=$(basename $yaml_file)
    model=${model%.*}

    # Ignore if it starts with _
    if [[ $model == _* ]]; then
        continue
    fi

    # Run for 5 seeds
    for seed in {0..4}; do
        echo "Running $model with seed $seed"
        mil_env/bin/python train.py +selected_model/$selected_model_type=$model experiment=$dataset group=$group_prefix-$dataset-$model seed=$seed
    done
done