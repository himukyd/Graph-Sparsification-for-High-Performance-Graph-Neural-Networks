#!/bin/bash
#SBATCH --job-name=gcn_speed_flexible
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output/gcn_flex_%j.out
#SBATCH --error=output/gcn_flex_%j.err

module purge
module load cuda-12.1
source /data/himanshu/env_him/bin/activate

EPOCHS=50
DATASET="reddit"

echo "--- STARTING FLEXIBLE GCN SPEED COMPARISON ---"

# 1. Baseline: ORIGINAL Graph
echo "Step 1: Training on Original Graph..."
python gcn_node_classification.py --dataset $DATASET --train_graph original --epoch $EPOCHS > ./output/gcn_original_${SLURM_JOB_ID}.txt

# 2. Loop through K values
K_VALUES=(493)

for K in "${K_VALUES[@]}"
do
    SPARSE_GRAPH="${DATASET}_local_top${K}_eigen.dgl"
    
    echo "----------------------------------------------------------"
    echo "PROCESSING K=$K"
    echo "----------------------------------------------------------"
    
    # Check if the sparsified graph exists, if not generate it
    if [ ! -f "$SPARSE_GRAPH" ]; then
        echo "--> Graph $SPARSE_GRAPH not found. Generating now..."
        python local_topk_sparsification.py --dataset $DATASET --k $K
    fi
    
    # Run GCN speed test
    if [ -f "$SPARSE_GRAPH" ]; then
        echo "--> Training GCN on $SPARSE_GRAPH..."
        python gcn_node_classification.py --dataset $DATASET --train_graph $SPARSE_GRAPH --epoch $EPOCHS > ./output/gcn_sparsified_K${K}_${SLURM_JOB_ID}.txt
        echo "--> Done. Results in ./output/gcn_sparsified_K${K}_${SLURM_JOB_ID}.txt"
    else
        echo "--> Error: Could not generate $SPARSE_GRAPH. Skipping."
    fi
done

echo "--- ALL GCN SPEED EXPERIMENTS COMPLETE ---"
