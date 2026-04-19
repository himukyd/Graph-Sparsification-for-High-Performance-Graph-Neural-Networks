import os

# Set DGL backend and home directory BEFORE importing DGL
os.environ['DGLBACKEND'] = 'pytorch'
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), ".matplotlib")

# Redirect DGL config directory to a writable location
try:
    os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")
    os.makedirs(os.environ['DGL_HOME'], exist_ok=True)
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
except Exception:
    pass

import dgl
import torch
import numpy as np
import time
import argparse
import sys
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, RedditDataset, YelpDataset, PubmedGraphDataset, CoraGraphDataset, CiteseerGraphDataset
from dgl.data import FraudYelpDataset as LegacyFraudYelpDataset

def load_dataset(dataset_name, dataset_path):
    print(f"--> [LOG] Loading {dataset_name}...")
    if dataset_name.startswith("ogbn-"):
        dataset = DglNodePropPredDataset(name=dataset_name, root=dataset_path)
        g, labels = dataset[0]
        # Attach labels and masks for OGB
        g.ndata['label'] = labels.squeeze().long()
        split_idx = dataset.get_idx_split()
        num_nodes = g.num_nodes()
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True
        
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
    elif dataset_name == "yelp":
        dataset = LegacyFraudYelpDataset(raw_dir=dataset_path)
        g_raw = dataset[0]
        g = dgl.to_homogeneous(g_raw)
        g.ndata['feat'] = g_raw.ndata['feature']
        g.ndata['label'] = g_raw.ndata['label'].long()
        g.ndata['train_mask'] = g_raw.ndata['train_mask']
        g.ndata['val_mask'] = g_raw.ndata['val_mask']
        g.ndata['test_mask'] = g_raw.ndata['test_mask']
    elif dataset_name == "reddit":
        dataset = RedditDataset()
        g = dataset[0] #standard built-in DGL dataset
    elif dataset_name == "pubmed":
        dataset = PubmedGraphDataset()
        g = dataset[0]
    elif dataset_name == "cora":
        dataset = CoraGraphDataset()
        g = dataset[0]
    elif dataset_name == "citeseer":
        dataset = CiteseerGraphDataset()
        g = dataset[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Ensure graph is simple and bidirected for eigenvector calculation
    g = dgl.to_simple(g) ## No self loop and no parallel edges 
    g = dgl.to_bidirected(g, copy_ndata=True) ## if graph is directed then convert it into bidirectional
    return g

def compute_eigenvector_centrality(g, device, max_iter=100, tol=1e-6):
    print(f"--> [LOG] Computing Eigenvector Centrality on {device}...")
    num_nodes = g.num_nodes()
    # x = torch.randn(num_nodes, 1).to(device)
    x = torch.ones((num_nodes, 1), device=device)
    x = x / torch.norm(x) # Normalize 

    # Use DGL's optimized message passing for Power Iteration: x = A * x
    g = g.to(device)
    
    start_time = time.time()
    for i in range(max_iter):
        x_old = x.clone()
        # Message passing: sum neighbor features
        g.ndata['h'] = x
        g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
        x = g.ndata['h']
        
        # Normalize
        norm = torch.norm(x)
        if norm > 0:
            x = x / norm
        
        # Check convergence
        diff = torch.norm(x - x_old)
        if diff < tol:
            print(f"--> [LOG] Converged at iteration {i+1}")
            break
    
    end_time = time.time()
    print(f"--> [LOG] Eigenvector calculation took {end_time - start_time:.4f}s")
    return x.cpu().squeeze()

def sparsify_graph(g, x, keep_fraction, ensure_connectivity=True):
    print(f"--> [LOG] Calculating edge scores and sparsifying using Eigenvector strategy...")
    u, v = g.edges() # u = source nodes, v = destination nodes
    u_np = u.numpy() # source nodes
    v_np = v.numpy() # destination nodes
    x_np = x.numpy() # node score 

    # Score e_ij = x_i * x_j (Eigenvector Centrality product)
    scores = x_np[u_np] * x_np[v_np]
    
    # Sort edges by score descending
    num_edges = len(scores)
    idx = np.argsort(-scores) # Negative for descending
    
    target_edges = int(num_edges * keep_fraction)
    print(f"--> [LOG] Target edge count: {target_edges} / {num_edges}")

    if ensure_connectivity:
        print(f"--> [LOG] Ensuring connectivity (Maximum Spanning Tree + Top Edges)...")
        # Union-Find to keep track of components
        parent = np.arange(g.num_nodes())
        def find(i):
            root = i
            while parent[root] != root:
                root = parent[root]
            # Path compression
            while parent[i] != root:
                next_node = parent[i]
                parent[i] = root
                i = next_node
            return root

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                return True
            return False

        keep_mask = np.zeros(num_edges, dtype=bool)
        edges_added = 0
        
        # 1. Add edges to maintain original components (Maximum Spanning Tree approach)
        for i in idx:
            node_u, node_v = u_np[i], v_np[i]
            if union(node_u, node_v):
                keep_mask[i] = True
                edges_added += 1
        
        print(f"--> [LOG] Added {edges_added} edges to maintain connectivity.")

        # 2. Add remaining top edges until we reach target_edges
        remaining_needed = target_edges - edges_added
        if remaining_needed > 0:
            # Mask out already added edges and pick top remaining
            top_indices = idx[~keep_mask[idx]][:remaining_needed]
            keep_mask[top_indices] = True
            edges_added += len(top_indices)
                
        print(f"--> [LOG] Total edges kept: {edges_added}")
    else:
        # Simple thresholding
        keep_mask = np.zeros(num_edges, dtype=bool)
        keep_mask[idx[:target_edges]] = True

    # Create new graph
    u_new = u[keep_mask]
    v_new = v[keep_mask]
    g_new = dgl.graph((u_new, v_new), num_nodes=g.num_nodes())
    
    # Copy node data
    for key, val in g.ndata.items():
        g_new.ndata[key] = val
        
    return g_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--path", default="/data/dgl_lab")
    parser.add_argument("--keep_fraction", type=float, default=0.5, help="Fraction of edges to keep")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    g = load_dataset(args.dataset, args.path)
    print(f"Original Graph: {g}")

    scores_node = compute_eigenvector_centrality(g, args.device, max_iter=args.max_iter)
    
    # Move graph to CPU for sparsification to ensure numpy compatibility
    g = g.to('cpu')
    
    g_sparsified = sparsify_graph(g, scores_node, args.keep_fraction, ensure_connectivity=True)
    
    # Save the sparsified graph
    output_name = f"{args.dataset}_eigen_p{args.keep_fraction}.dgl"
    dgl.save_graphs(output_name, [g_sparsified])
    print(f"--> [SUCCESS] Sparsified graph saved to {output_name}")

    print("\n" + "="*40)
    print("      GRAPH SPARSIFICATION SUMMARY")
    print("="*40)
    print(f"Original Graph:  Nodes: {g.num_nodes():<10} Edges: {g.num_edges() // 2:<10}")
    print(f"Sparsified Graph: Nodes: {g_sparsified.num_nodes():<10} Edges: {g_sparsified.num_edges() // 2:<10}")
    print(f"Retention:       {args.keep_fraction*100:.1f}% of edges kept")
    print("="*40 + "\n")
