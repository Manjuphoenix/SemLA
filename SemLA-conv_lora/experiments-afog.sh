#!/bin/bash

# export DETECTRON2_DATASETS="/data/aditya/ov-seg-main/datasets/"
# export DETECTRON2_DATASETS="/datasets/"


semla_config='./merge_confs/afog/semla_config.yaml'
source_domains='./merge_confs/afog/source_domains.yaml'
target_domains='./merge_confs/afog/target_domains.yaml'
results_folder='./merge_confs/afog/results/semla'

# Uncomment the experiments you want to run

# python experiments.py --experiment zeroshot \
#     --source_domains $source_domains \
#     --target_domains $target_domains \
#     --output_dir $results_folder

# python experiments.py --experiment oracle \
#     --source_domains $source_domains \
#     --target_domains $target_domains \
#     --output_dir $results_folder

# python experiments.py --experiment uniform \
#     --source_domains $source_domains \
#     --target_domains $target_domains \
#     --remove_target_adapter \
#     --output_dir $results_folder

python experiments.py --experiment semla \
    --source_domains $source_domains \
    --target_domains $target_domains \
    --semla_config $semla_config \
    --remove_target_adapter \
    --output_dir $results_folder