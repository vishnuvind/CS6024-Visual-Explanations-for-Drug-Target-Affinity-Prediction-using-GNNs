#!/bin/bash

datasets=("kiba")        # set dataset (only Davis/KIBA supported)
gnns=("gat")

for dataset in ${datasets[@]}; do
    for gnn in ${gnns[@]}; do
        python3 train.py --model=$gnn --dataset=$dataset
    done
done