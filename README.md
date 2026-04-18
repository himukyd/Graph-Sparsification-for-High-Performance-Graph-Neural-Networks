# Graph Sparsification Benchmarks for Scalable GNNs

This project implements and evaluates advanced **Graph Sparsification** techniques to address the computational bottlenecks associated with training Graph Neural Networks (GNNs) on large-scale datasets. By strategically removing redundant or less informative edges, we aim to reduce memory footprint and execution time while maintaining high classification accuracy.

---

##  Project Objectives

As graph datasets (like OGBN-Products or Reddit) grow to millions of nodes and edges, standard GCN architectures face significant scalability challenges. This project explores three distinct sparsification strategies to find the optimal balance between **graph density** and **model performance**:

1.  **Spectral (Eigen) Sparsification:** Uses Eigenvector Centrality and Power Iteration to identify and preserve the structural "backbone" of the graph.
2.  **Local Top-K Sparsification:** A node-centric approach that ensures local connectivity is preserved by keeping the most significant edges for every individual node.
3.  **Random Sparsification:** A stochastic baseline to quantify the performance gain of informed sparsification methods over simple uniform edge removal.

---

## 🛠️ Methodologies

### 1. Eigen-based Spectral Sparsification
We implement **Eigenvector Centrality** using an optimized **Power Iteration** method on the GPU. The algorithm identifies nodes that are most influential within the network's global structure. Edges connected to high-centrality nodes are prioritized, ensuring that the spectral properties of the original graph are approximated in the sparsified version.

### 2. Local Top-K Selection
To maintain local neighborhood information, this method iterates through each node and selects the top $K$ edges based on connectivity or importance. This ensures that no node becomes completely isolated and that the "local context" required for GCN message passing remains intact.

### 3. GCN Classification Pipeline
The project utilizes a standard **Graph Convolutional Network (GCN)** architecture implemented in PyTorch and DGL. The pipeline involves:
- Loading large-scale datasets (OGB, Reddit, Yelp, IGB).
- Applying a sparsification transformation.
- Training the GCN on the sparsified adjacency matrix.
- Evaluating accuracy, convergence speed, and memory usage against the full-graph baseline.

---

## 🖥️ Experimental Environment (HPC)

All benchmarks were performed on a **High-Performance Computing (HPC)** cluster to ensure consistent and reproducible results.

### Hardware Stack
- **Node:** Dedicated compute node `gpu1`.
- **CPU:** Intel(R) Xeon(R) Gold 5218 (16 Cores / 32 Threads, 2.30 GHz).
- **GPU:** NVIDIA RTX A6000 (48 GB GDDR6 VRAM).
- **Architecture:** x86_64 with AVX-512 support for accelerated tensor operations.

### Software Stack
- **OS:** Linux (accessed via SSH).
- **Language:** Python 3.11.4.
- **Deep Learning:** PyTorch 2.3.0 (CUDA 12.1).
- **Graph Framework:** DGL 2.5.0 (Deep Graph Library).
- **Drivers:** NVIDIA Driver 550.54.14 / CUDA 12.4.

---

##  Getting Started

### Prerequisites
Ensure your environment matches the HPC specifications above. You can set up the virtual environment using:

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install DGL with CUDA support
pip install dgl==2.5.0+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html

# Additional dependencies
pip install numpy ogb
```

### Execution
The project is modular. You can run sparsification and training separately to inspect intermediate results.

#### Step 1: Sparsify the Graph
```bash
# Perform Eigen-based sparsification (p = keep probability)
python eigen_sparsification.py --dataset ogbn-arxiv --p 0.7
```

#### Step 2: Train GCN on Sparsified Data
```bash
# Train on the sparsified .dgl file
python gcn_node_classification.py --dataset ogbn-arxiv --gpu 0
```

---

##  Evaluation Metrics
We measure the success of our sparsification methods using three primary KPIs:
- **Accuracy Retention:** Percentage of baseline accuracy maintained after $X\%$ edge removal.
- **Training Speedup:** Reduction in time per epoch during the GNN training phase.
- **Memory Efficiency:** Reduction in peak GPU memory usage during message passing.
