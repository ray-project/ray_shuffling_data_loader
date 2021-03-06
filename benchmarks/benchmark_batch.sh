#!/bin/bash

# Exit immediately if any command fails.
set -exo pipefail

data_dir="s3://shuffling-data-loader-benchmarks/data/"
stats_dir="s3://shuffling-data-loader-benchmarks/stats/"

num_rows=$((4 * (10 ** 8)))
num_row_groups_per_file=5
batch_size=250000
num_trials=2
num_epochs=10

max_concurrent_epochs_list=(2)
num_files_list=(100 50 25)
num_trainers_list=(16 8 4)
num_reducers_per_trainer_list=(4 3 2)

for max_concurrent_epochs in "${max_concurrent_epochs_list[@]}"; do
        for num_files in "${num_files_list[@]}"; do
                for num_trainers in "${num_trainers_list[@]}"; do
                        for num_reducers_per_trainer in "${num_reducers_per_trainer_list[@]}"; do
                                num_reducers=$(( num_reducers_per_trainer * num_trainers ))
                                ray exec cluster.yaml "python ~/benchmarks/benchmark.py \
                                        --num-rows $num_rows \
                                        --num-files $num_files \
                                        --num-row-groups-per-file $num_row_groups_per_file \
                                        --batch-size $batch_size \
                                        --num-trials $num_trials \
                                        --cluster \
                                        --num-reducers $num_reducers \
                                        --num-trainers $num_trainers \
                                        --num-epochs $num_epochs \
                                        --max-concurrent-epochs $max_concurrent_epochs \
                                        --data-dir $data_dir \
                                        --stats-dir $stats_dir \
                                        --unique-stats"
                                done
                        done
                done
        done
