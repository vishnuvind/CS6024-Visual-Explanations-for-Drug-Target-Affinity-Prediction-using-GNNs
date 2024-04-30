#!/bin/bash

dataset="davis"        # set dataset (only Davis/KIBA supported)
gnns=("gcn" "gin" "gat")

gcn_davis='./save/20240428_152548_davis_gcn/model/model-gcn, dataset-davis, epoch-461, loss-0.0736, cindex-0.9349, test_loss-0.2135.pt'
gin_davis='./save/20240428_162841_davis_gin/model/model-gin, dataset-davis, epoch-497, loss-0.1063, cindex-0.9175, test_loss-0.2244.pt'
gat_davis='./save/20240428_172725_davis_gat/model/model-gat, dataset-davis, epoch-476, loss-0.1162, cindex-0.9148, test_loss-0.2225.pt'
# threshs=(-1 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
threshs=(-1 0.50)

# # compute baseline performance scores
# python3 explain_drug_target.py --model='gcn' --dataset='davis' --saved_model="$gcn_davis" --thresh=0.0
# python3 explain_drug_target.py --model='gin' --dataset='davis' --saved_model="$gin_davis" --thresh=0.0
# python3 explain_drug_target.py --model='gat' --dataset='davis' --saved_model="$gat_davis" --thresh=0.0

# # compute soft thresholding performance scores
# python3 explain_drug_target.py --model='gcn' --dataset='davis' --saved_model="$gcn_davis" --mask_drug
# python3 explain_drug_target.py --model='gin' --dataset='davis' --saved_model="$gin_davis" --mask_drug
# python3 explain_drug_target.py --model='gat' --dataset='davis' --saved_model="$gat_davis" --mask_drug

for thresh in ${threshs[@]}; do
    # python3 explain_drug_target.py --model='gcn' --dataset='davis' --saved_model="$gcn_davis" --thresh=$thresh --mask_drug
    # python3 explain_drug_target.py --model='gcn' --dataset='davis' --saved_model="$gcn_davis" --thresh=$thresh --mask_targ
    python3 explain_drug_target.py --model='gcn' --dataset='davis' --saved_model="$gcn_davis" --thresh=$thresh --mask_drug --mask_targ
    
    # python3 explain_drug_target.py --model='gin' --dataset='davis' --saved_model="$gin_davis" --thresh=$thresh --mask_drug
    # python3 explain_drug_target.py --model='gin' --dataset='davis' --saved_model="$gin_davis" --thresh=$thresh --mask_targ
    # python3 explain_drug_target.py --model='gin' --dataset='davis' --saved_model="$gin_davis" --thresh=$thresh --mask_drug --mask_targ
    
    # python3 explain_drug_target.py --model='gat' --dataset='davis' --saved_model="$gat_davis" --thresh=$thresh --mask_drug
    # python3 explain_drug_target.py --model='gat' --dataset='davis' --saved_model="$gat_davis" --thresh=$thresh --mask_targ
    # python3 explain_drug_target.py --model='gat' --dataset='davis' --saved_model="$gat_davis" --thresh=$thresh --mask_drug --mask_targ
done