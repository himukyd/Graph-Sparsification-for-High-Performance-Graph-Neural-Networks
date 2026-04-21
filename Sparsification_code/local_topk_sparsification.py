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
from dgl.data import RedditDataset, YelpDataset, PubmedGraphDataset
from dgl.data import FraudYelpDataset as LegacyFraudYelpDataset

def load_dataset(dataset_name, dataset_path):
    print(f"--> [LOG] Loading {dataset_name}...")
    if dataset_name.startswith("ogbn-"):
        dataset = DglNodePropPredDataset(name=dataset_name, root=dataset_path)
        g, labels = dataset[0]
        # OGB labels are a dictionary, extract the labels tensor
        g.ndata['label'] = labels.squeeze().long()
        # OGB datasets already have train/val/test masks in the split_idx
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
        g = dataset[0]
    elif dataset_name == "pubmed":
        dataset = PubmedGraphDataset()
        g = dataset[0]
    elif dataset_name == "igb-small":
        try:
            from igb.dataloader import IGBHomogeneousDGLDataset
            dataset = IGBHomogeneousDGLDataset(name='small', root=dataset_path, in_memory=1)
            g = dataset[0]
        except ImportError:
            from dgl.data import IGBDataset
            dataset = IGBDataset(name='igb-hom-small', root=dataset_path)
            g = dataset[0]
    elif dataset_name == "igb-medium":
        try:
            from igb.dataloader import IGBHomogeneousDGLDataset
            dataset = IGBHomogeneousDGLDataset(name='medium', root=dataset_path, in_memory=1)
            g = dataset[0]
        except ImportError:
            from dgl.data import IGBDataset
            dataset = IGBDataset(name='igb-hom-medium', root=dataset_path)
            g = dataset[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Ensure graph is simple and bidirected for eigenvector calculation
    g = dgl.to_simple(g)
    g = dgl.to_bidirected(g, copy_ndata=True)
    return g

def compute_eigenvector_centrality(g, device, max_iter=100, tol=1e-6):
    print(f"--> [LOG] Computing Eigenvector Centrality on {device}...")
    num_nodes = g.num_nodes()
    x = torch.ones((num_nodes, 1), device=device)
    x = x / torch.norm(x)
    g = g.to(device)
    
    start_time = time.time()
    for i in range(max_iter):
        x_old = x.clone()
        g.ndata['h'] = x
        g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
        x = g.ndata['h']
        
        norm = torch.norm(x)
        if norm > 0:
            x = x / norm
        
        if torch.norm(x - x_old) < tol:
            print(f"--> [LOG] Converged at iteration {i+1}")
            break
    
    end_time = time.time()
    print(f"--> [LOG] Eigenvector calculation took {end_time - start_time:.4f}s")
    return x.cpu().squeeze()

def local_topk_sparsify(g, x, k_neighbors):
    print(f"--> [LOG] Applying Local Top-{k_neighbors} Sparsification...")
    num_nodes = g.num_nodes()
    u, v = g.edges()
    u_np = u.numpy()
    v_np = v.numpy()
    x_np = x.numpy()

    # Calculate all edge scores: Score = x_i * x_j
    scores = x_np[u_np] * x_np[v_np]
    
    # Efficiently select top K neighbors for each node
    keep_edge_indices = set()
    row_ptr, _, _ = g.adj_tensors('csr')

    for i in range(num_nodes):
        start = row_ptr[i].item()
        end = row_ptr[i+1].item()
        
        if start == end: 
            continue # Skip isolated nodes
        
        node_edge_scores = scores[start:end]
        node_edge_indices = np.arange(start, end)
        
        # Take all neighbors if count < K, else take top K
        num_to_keep = min(end - start, k_neighbors)
        top_idx = np.argsort(node_edge_scores)[-num_to_keep:]
        
        for idx in top_idx:
            keep_edge_indices.add(node_edge_indices[idx])

    # Construct the final sparsified graph
    keep_list = sorted(list(keep_edge_indices))
    g_new = dgl.graph((u[keep_list], v[keep_list]), num_nodes=num_nodes)
    
    for key, val in g.ndata.items():
        g_new.ndata[key] = val
        
    return g_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--path", default="/data/dgl_lab")
    parser.add_argument("--k", type=int, default=10, help="Number of top neighbors to keep per node")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    g = load_dataset(args.dataset, args.path)
    print(f"Original Graph: {g}")

    scores_node = compute_eigenvector_centrality(g, args.device)
    
    g_sparsified = local_topk_sparsify(g, scores_node, args.k)
    
    # Save results
    output_name = f"{args.dataset}_local_top{args.k}_eigen.dgl"
    dgl.save_graphs(output_name, [g_sparsified])
    
    print("\n" + "="*40)
    print(f"      LOCAL TOP-{args.k} SUMMARY")
    print("="*40)
    print(f"Original Graph:  Nodes: {g.num_nodes():<10} Edges: {g.num_edges() // 2:<10}")
    print(f"Sparsified Graph: Nodes: {g_sparsified.num_nodes():<10} Edges: {g_sparsified.num_edges() // 2:<10}")
    print(f"Strategy:        Each node kept up to {args.k} neighbors based on Eigen Centrality.")
    print("="*40 + "\n")
