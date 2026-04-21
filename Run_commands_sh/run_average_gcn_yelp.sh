#!/bin/bash
#SBATCH --job-name=gcn_yelp_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output_4/gcn_yelp_%j.out
#SBATCH --error=output_4/gcn_yelp_%j.err

module purge
module load cuda-12.1
source /data/himanshu/env_him/bin/activate

DATASET="yelp"
EPOCHS=50

echo "--- STARTING YELP EXPERIMENT ---"

# 1. Baseline: ORIGINAL Graph
echo "Step 1: Training GCN on Original Graph..."
python gcn_node_classification.py --dataset $DATASET --train_graph original --epoch $EPOCHS > ./output_4/yelp_original_${SLURM_JOB_ID}.txt

# 2. Local Top-K Sparsification
K_VALS=(168)

for K in "${K_VALS[@]}"
do
    SPARSE_GRAPH="${DATASET}_local_top${K}_eigen.dgl"
    
    echo "----------------------------------------------------------"
    echo "PROCESSING K=$K"
    echo "----------------------------------------------------------"
    
    # Generate Sparsified Graph
    echo "--> Generating $SPARSE_GRAPH..."
    python local_topk_sparsification.py --dataset $DATASET --k $K
    
    # Run GCN speed test on Sparsified Graph
    if [ -f "$SPARSE_GRAPH" ]; then
        echo "--> Training GCN on $SPARSE_GRAPH..."
        python gcn_node_classification.py --dataset $DATASET --train_graph $SPARSE_GRAPH --epoch $EPOCHS > ./output_4/yelp_local_top${K}_${SLURM_JOB_ID}.txt
    else
        echo "--> Error: Could not generate $SPARSE_GRAPH."
    fi
done

echo "--- YELP EXPERIMENT COMPLETE ---"
