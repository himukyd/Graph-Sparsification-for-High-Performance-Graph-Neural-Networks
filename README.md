# Graph Sparsification Benchmarks for Scalable GNNs

This repository benchmarks graph sparsification methods for improving training efficiency of Graph Neural Networks (GNNs) while preserving predictive quality.

## Overview

The project focuses on two sparsification strategies:

1. Eigen-based global sparsification
2. Local Top-K sparsification

Both methods produce sparsified DGL graphs that are then used in a GCN training pipeline for node classification benchmarks.

## Repository Structure

- `Sparsification_code/`
	- `eigen_sparsification.py`: global spectral-style sparsification using eigenvector centrality scores.
	- `local_topk_sparsification.py`: per-node local top-k edge selection using centrality-derived scores.
- `GCN_code/`
	- `gcn_node_classification.py`: GCN training and evaluation on original or sparsified graphs.
- `Run_commands_sh/`
	- SLURM scripts for dataset-wise experiment batches.
- `Plots/`
	- Plot scripts for edge-accuracy and sparsification visualizations.

## Supported Datasets

- `reddit`
- `yelp`
- `ogbn-arxiv`
- `ogbn-products`
- `pubmed`
- `igb-small`, `igb-medium` (if IGB package/dataset is available)

## Environment Setup

Example Python setup (CUDA 12.1 stack):

```bash
# PyTorch
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# DGL with CUDA
pip install dgl==2.5.0+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html

# Other dependencies
pip install numpy ogb torchmetrics matplotlib
```

Notes:

- Scripts default to dataset root path `--path /data/dgl_lab` (HPC-oriented default).
- Override with your own data root when running locally.

## Quick Start (Manual)

Run commands from repository root.

### 1) Create a sparsified graph

Global eigen sparsification:

```bash
python Sparsification_code/eigen_sparsification.py \
	--dataset ogbn-arxiv \
	--keep_fraction 0.7 \
	--path /data/dgl_lab
```

Local Top-K sparsification:

```bash
python Sparsification_code/local_topk_sparsification.py \
	--dataset ogbn-arxiv \
	--k 50 \
	--path /data/dgl_lab
```

### 2) Train GCN on original graph

```bash
python GCN_code/gcn_node_classification.py \
	--dataset ogbn-arxiv \
	--train_graph original \
	--epoch 50 \
	--path /data/dgl_lab
```

### 3) Train GCN on sparsified graph

```bash
python GCN_code/gcn_node_classification.py \
	--dataset ogbn-arxiv \
	--train_graph ogbn-arxiv_local_top50_eigen.dgl \
	--epoch 50 \
	--path /data/dgl_lab
```

## Batch Runs (SLURM)

Use scripts inside `Run_commands_sh/` to reproduce larger sweeps.

Examples:

```bash
sbatch Run_commands_sh/run_1_gcn_reddit.sh
sbatch Run_commands_sh/run_2_gcn_product.sh
sbatch Run_commands_sh/run_3_gcn_arxiv.sh
sbatch Run_commands_sh/run_4_gcn_yelp.sh
```

Global eigen sweeps:

```bash
sbatch Run_commands_sh/run_reddit_eigen_global.sh
sbatch Run_commands_sh/run_arxiv_eigen_global.sh
sbatch Run_commands_sh/run_product_eigen_global.sh
sbatch Run_commands_sh/run_yelp_eigen_global.sh
```

## Evaluation Metrics

The core metrics used for comparison are:

- Accuracy retention: relative model quality vs original graph baseline.
- Training time: epoch-level and total runtime for GCN training.
- Edge reduction: final sparsified edge count vs original edge count.

## Final Results

The table below reports the best run for each method family on each dataset, extracted from experiment logs in `output/`, `output_2/`, `output_3/`, and `output_4/`.

| Dataset | Method | Best Config | Best Val | Best Test | Total Time (s) | Delta Test vs Baseline | Speedup vs Baseline |
|---|---|---|---:|---:|---:|---:|---:|
| reddit | baseline | original | 0.9410 | 0.9374 | 2656.4248 | +0.0000 | 1.00x |
| reddit | local-topk | K=1000 | 0.9417 | 0.9385 | 1744.4921 | +0.0011 | 1.52x |
| reddit | eigen-global | p=0.8 | 0.8995 | 0.8983 | 2532.6214 | -0.0391 | 1.05x |
| ogbn-arxiv | baseline | original | 0.7086 | 0.7006 | 60.4754 | +0.0000 | 1.00x |
| ogbn-arxiv | local-topk | K=250 | 0.7066 | 0.6962 | 48.7224 | -0.0044 | 1.24x |
| ogbn-arxiv | eigen-global | p=0.8 | 0.6680 | 0.6602 | 59.1732 | -0.0404 | 1.02x |
| ogbn-products | baseline | original | 0.9134 | 0.7346 | 3015.1128 | +0.0000 | 1.00x |
| ogbn-products | local-topk | K=500 | 0.9133 | 0.7270 | 3179.8030 | -0.0076 | 0.95x |
| ogbn-products | eigen-global | p=0.8 | 0.8974 | 0.6892 | 2602.7670 | -0.0454 | 1.16x |
| yelp | baseline | original | 0.8590 | 0.8588 | 26.1353 | +0.0000 | 1.00x |
| yelp | local-topk | K=5 | 0.8742 | 0.8732 | 12.6630 | +0.0144 | 2.06x |
| yelp | eigen-global | p=0.5 | 0.8585 | 0.8589 | 18.1751 | +0.0001 | 1.44x |

Key observations:

- Local Top-K gave the strongest accuracy-time tradeoff on reddit and yelp.
- On ogbn-arxiv, Local Top-K was close to baseline accuracy with a useful runtime reduction.
- On ogbn-products, aggressive sparsification reduced runtime for eigen-global but with larger accuracy drop; local-topk preserved accuracy better but the best-accuracy local setting did not improve runtime.
- Eigen-global was generally more conservative in speed gains except on yelp/products where modest speedups were observed.

These values reflect the currently stored logs and can change if rerun with different seeds, epochs, or hardware.

## Reproducibility Notes

- Output logs are written under `output/`, `output_2/`, `output_3/`, `output_4/`, and `output_5/` by the SLURM scripts.
- Pre-generated `.dgl` sparsified graphs are included in this repository for common datasets/settings.
- For strict reproducibility, keep software versions aligned with the setup section.
