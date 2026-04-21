#!/bin/bash
#SBATCH --job-name=ogbn_arxiv_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output_3/ogbn_arxiv_%j.out
#SBATCH --error=output_3/ogbn_arxiv_%j.err

module purge
module load cuda-12.1
source /data/himanshu/env_him/bin/activate

DATASET="ogbn-arxiv"
EPOCHS=50

echo "--- STARTING OGBN-ARXIV EXPERIMENT ---"

# 1. Baseline: ORIGINAL Graph
echo "Step 1: Training GCN on Original Graph..."
python gcn_node_classification.py --dataset $DATASET --train_graph original --epoch $EPOCHS > ./output_3/arxiv_original_${SLURM_JOB_ID}.txt

# 2. Local Top-K Sparsification
K_VALS=(15)

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
        python gcn_node_classification.py --dataset $DATASET --train_graph $SPARSE_GRAPH --epoch $EPOCHS > ./output_3/arxiv_local_top${K}_${SLURM_JOB_ID}.txt
    else
        echo "--> Error: Could not generate $SPARSE_GRAPH."
    fi
done

echo "--- OGBN-ARXIV EXPERIMENT COMPLETE ---"
