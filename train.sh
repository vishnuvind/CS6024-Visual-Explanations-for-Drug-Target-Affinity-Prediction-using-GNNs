#!/bin/bash

datasets=("davis" "kiba")        # set dataset (only Davis/KIBA supported)
gnns=("gcn" "gin" "gat")

for dataset in ${datasets[@]}; do
    for gnn in ${gnns[@]}; do
        python3 train.py --model=$gnn --dataset=$dataset
    done
done