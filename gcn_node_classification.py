import os

# Set DGL backend and home directory BEFORE importing DGL
os.environ['DGLBACKEND'] = 'pytorch'
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), ".matplotlib")
os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")

# Ensure writable directories exist
try:
    os.makedirs(os.environ['DGL_HOME'], exist_ok=True)
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
except Exception:
    pass

import argparse
import time
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
import sys
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import RedditDataset, YelpDataset, PubmedGraphDataset

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(dglnn.GraphConv(in_size, out_size, allow_zero_in_degree=True))
        else:
            self.layers.append(dglnn.GraphConv(in_size, hid_size, allow_zero_in_degree=True))
            for _ in range(num_layers - 2):
                self.layers.append(dglnn.GraphConv(hid_size, hid_size, allow_zero_in_degree=True))
            self.layers.append(dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

def evaluate(model, graph, dataloader, num_classes, is_multilabel):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    
    y_hats_cat = torch.cat(y_hats)
    ys_cat = torch.cat(ys)

    if is_multilabel:
        preds = torch.sigmoid(y_hats_cat)
        return MF.f1_score(preds, ys_cat.int(), task="multilabel", num_labels=num_classes, average="macro")
    else:
        return MF.accuracy(y_hats_cat, ys_cat, task="multiclass", num_classes=num_classes)

def train(args, device, g_train, model, num_classes, is_multilabel):
    train_idx = torch.nonzero(g_train.ndata['train_mask']).squeeze()
    val_idx = torch.nonzero(g_train.ndata['val_mask']).squeeze()
    test_idx = torch.nonzero(g_train.ndata['test_mask']).squeeze()

    sampler = NeighborSampler(
        [-1] * args.num_layers,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    
    train_dataloader = DataLoader(
        g_train, train_idx.to(device), sampler,
        device=device, batch_size=args.batch_size, shuffle=True
    )

    val_dataloader = DataLoader(
        g_train, val_idx.to(device), sampler,
        device=device, batch_size=args.batch_size, shuffle=False
    )

    test_dataloader = DataLoader(
        g_train, test_idx.to(device), sampler,
        device=device, batch_size=args.batch_size, shuffle=False
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best_val_acc = 0
    best_test_acc = 0
    
    print(f"Starting training on {device}...")
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            
            if is_multilabel:
                loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
            else:
                loss = F.cross_entropy(y_hat, y.long())
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        val_acc = evaluate(model, g_train, val_dataloader, num_classes, is_multilabel)
        test_acc = evaluate(model, g_train, test_dataloader, num_classes, is_multilabel)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        print(f"Epoch {epoch:03d} | Loss {total_loss/(it+1):.4f} | Val Acc: {val_acc.item():.4f} | Test Acc: {test_acc.item():.4f} | Time: {epoch_time:.4f}s")
        
    return best_val_acc, best_test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--train_graph", type=str, default="original")
    parser.add_argument('--path', type=str, default='/data/dgl_lab')
    args = parser.parse_args()

    # Data Loading
    if args.dataset == "reddit":
        dataset = RedditDataset()
        g = dataset[0]
        num_classes = dataset.num_classes
    elif args.dataset.startswith("ogbn-"):
        dataset = DglNodePropPredDataset(name=args.dataset, root=args.path)
        g, labels = dataset[0]
        # Attach labels and masks for OGB
        g.ndata['label'] = labels.squeeze().long()
        split_idx = dataset.get_idx_split()
        g.ndata['train_mask'] = torch.zeros(g.num_nodes(), dtype=torch.bool)
        g.ndata['val_mask'] = torch.zeros(g.num_nodes(), dtype=torch.bool)
        g.ndata['test_mask'] = torch.zeros(g.num_nodes(), dtype=torch.bool)
        g.ndata['train_mask'][split_idx['train']] = True
        g.ndata['val_mask'][split_idx['valid']] = True
        g.ndata['test_mask'][split_idx['test']] = True
        num_classes = dataset.num_classes
    elif args.dataset == "yelp":
        from dgl.data import FraudYelpDataset as LegacyFraudYelpDataset
        dataset = LegacyFraudYelpDataset(raw_dir=args.path)
        g_raw = dataset[0]
        g = dgl.to_homogeneous(g_raw)
        g.ndata['feat'] = g_raw.ndata['feature']
        g.ndata['label'] = g_raw.ndata['label'].long()
        g.ndata['train_mask'] = g_raw.ndata['train_mask']
        g.ndata['val_mask'] = g_raw.ndata['val_mask']
        g.ndata['test_mask'] = g_raw.ndata['test_mask']
        num_classes = dataset.num_classes
    elif args.dataset == "igb-small":
        try:
            from igb.dataloader import IGBHomogeneousDGLDataset
            dataset = IGBHomogeneousDGLDataset(name='small', root=args.path, in_memory=1)
            g = dataset[0]
            num_classes = dataset.num_classes
        except ImportError:
            from dgl.data import IGBDataset
            dataset = IGBDataset(name='igb-hom-small', root=args.path)
            g = dataset[0]
            num_classes = dataset.num_classes
    elif args.dataset == "igb-medium":
        try:
            from igb.dataloader import IGBHomogeneousDGLDataset
            dataset = IGBHomogeneousDGLDataset(name='medium', root=args.path, in_memory=1)
            g = dataset[0]
            num_classes = dataset.num_classes
        except ImportError:
            from dgl.data import IGBDataset
            dataset = IGBDataset(name='igb-hom-medium', root=args.path)
            g = dataset[0]
            num_classes = dataset.num_classes
    elif args.dataset == "pubmed":
        from dgl.data import PubmedGraphDataset
        dataset = PubmedGraphDataset()
        g = dataset[0]
        num_classes = dataset.num_classes
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
    
    # Ensure graph is simple and bidirected (undirected) to match sparsification baseline
    g = dgl.to_simple(g)
    g = dgl.to_bidirected(g, copy_ndata=True)
    print("--> [LOG] Graph converted to simple and bidirected (undirected).")
    # Determine training graph
    if args.train_graph == "original":
        g_train = g
    else:
        graphs, _ = dgl.load_graphs(args.train_graph)
        g_train = graphs[0]

    # Clean and add self-loops
    g_train = dgl.remove_self_loop(g_train)
    g_train = dgl.add_self_loop(g_train)

    is_multilabel = g_train.ndata['label'].ndim > 1 and g_train.ndata['label'].shape[1] > 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_train = g_train.to(device)

    print("\n" + "="*40)
    print(f"GCN SPEED TEST: {args.train_graph}")
    print(f"Nodes: {g_train.num_nodes()} | Edges: {g_train.num_edges() // 2}")
    print("="*40)

    model = GCN(g_train.ndata["feat"].shape[1], 256, num_classes, args.num_layers).to(device)
    
    start_train = time.time()
    best_val, best_test = train(args, device, g_train, model, num_classes, is_multilabel)
    total_time = time.time() - start_train
    
    print(f"\nTOTAL TRAINING TIME: {total_time:.4f}s")
    print(f"BEST VAL ACCURACY: {best_val.item():.4f}")
    print(f"TEST ACCURACY (at best val): {best_test.item():.4f}")
