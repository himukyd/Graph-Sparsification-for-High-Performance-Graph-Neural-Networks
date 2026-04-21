#!/bin/bash
#SBATCH --job-name=product_eigen_global
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 # Products is large, needs more time
#SBATCH --output=output_2/product_global_%j.out
#SBATCH --error=output_2/product_global_%j.err

module purge
module load cuda-12.1
source /data/himanshu/env_him/bin/activate

DATASET="ogbn-products"
EPOCHS=50

echo "--- STARTING PRODUCTS GLOBAL EIGEN SPARSIFICATION EXPERIMENT ---"

# 1. Baseline: ORIGINAL Graph
echo "Step 1: Training GCN on Original Graph..."
python gcn_node_classification.py --dataset $DATASET --train_graph original --epoch $EPOCHS > ./output_2/product_original_global_${SLURM_JOB_ID}.txt

# 2. Global Eigen Sparsification
FRACTIONS=(0.8 0.7 0.6 0.5)

for P in "${FRACTIONS[@]}"
do
    SPARSE_GRAPH="${DATASET}_eigen_p${P}.dgl"
    
    echo "----------------------------------------------------------"
    echo "PROCESSING Keep Fraction: $P"
    echo "----------------------------------------------------------"
    
    # Generate Sparsified Graph
    if [ ! -f "$SPARSE_GRAPH" ]; then
        echo "--> Generating $SPARSE_GRAPH..."
        python eigen_sparsification.py --dataset $DATASET --keep_fraction $P
    fi
    
    # Run GCN training on Sparsified Graph
    if [ -f "$SPARSE_GRAPH" ]; then
        echo "--> Training GCN on $SPARSE_GRAPH..."
        python gcn_node_classification.py --dataset $DATASET --train_graph $SPARSE_GRAPH --epoch $EPOCHS > ./output_2/product_eigen_p${P}_${SLURM_JOB_ID}.txt
    else
        echo "--> Error: Could not generate $SPARSE_GRAPH."
    fi
done

echo "--- PRODUCTS GLOBAL EIGEN EXPERIMENT COMPLETE ---"
