"""
AttentiveFP-style GNN for BBB Permeability Prediction
Run this on Google Colab (GPU recommended).

Install:
    !pip install torch torch-geometric rdkit-pypi scikit-learn

Author: sakeermr | github.com/sakeermr/bbb-permeability-gnn
Paper:  Xiong et al. (2020) Pushing the Boundaries of Molecular Representation
        for Drug Discovery with the Graph Attention Mechanism. JACS.
"""

# ── Colab setup ──────────────────────────────────────────────────────────────
# !pip install torch torch-geometric rdkit-pypi scikit-learn seaborn shap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool
from rdkit import Chem
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# ── Constants ────────────────────────────────────────────────────────────────
ATOM_DIM = 9
BOND_DIM = 4


# ── Featurisation ─────────────────────────────────────────────────────────────

def atom_features(atom) -> list:
    """9-dim atom feature vector."""
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 10.0,
        atom.GetFormalCharge() / 5.0,
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs() / 8.0,
        int(atom.IsInRing()),
        atom.GetMass() / 200.0,
        int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3),
        int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
    ]


def bond_features(bond) -> list:
    """4-dim bond feature vector."""
    return [
        bond.GetBondTypeAsDouble() / 3.0,
        int(bond.GetIsAromatic()),
        int(bond.IsInRing()),
        int(bond.GetIsConjugated()),
    ]


def smiles_to_graph(smiles: str, label: int) -> Data | None:
    """Convert SMILES → PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()],
                     dtype=torch.float)
    edge_idx, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_idx  += [[i, j], [j, i]]
        edge_attr += [bf, bf]
    if not edge_idx:
        edge_idx  = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_DIM), dtype=torch.float)
    else:
        edge_idx  = torch.tensor(edge_idx,  dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, y=y)


# ── Model ────────────────────────────────────────────────────────────────────

class BBBGraphNet(nn.Module):
    """
    AttentiveFP-style Graph Attention Network for BBB prediction.

    Architecture:
        Input node features (9-dim)
            ↓
        GAT Layer 1  (64 hidden, 4 heads, concat → 256-dim)
            ↓  BatchNorm + ELU + Dropout
        GAT Layer 2  (64 hidden, 1 head → 64-dim)
            ↓  BatchNorm + ELU
        Global pooling (sum + mean, → 64-dim)
            ↓
        MLP: 64 → 128 → 64 → 1
            ↓
        Sigmoid → P(BBB permeable)
    """
    def __init__(self, in_ch: int = ATOM_DIM, hidden: int = 64,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_ch,        hidden, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden*heads, hidden, heads=1,     dropout=dropout, concat=False)
        self.bn1  = nn.BatchNorm1d(hidden * heads)
        self.bn2  = nn.BatchNorm1d(hidden)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.bn1(self.gat1(x, ei)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.gat2(x, ei)))
        x = global_add_pool(x, batch) + global_mean_pool(x, batch)
        return torch.sigmoid(self.mlp(x)).squeeze()


# ── Training utilities ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.extend(model(batch).cpu().numpy().flatten())
        labels.extend(batch.y.cpu().numpy().flatten())
    preds, labels = np.array(preds), np.array(labels)
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, (preds >= 0.5).astype(int))
    f1  = f1_score(labels, (preds >= 0.5).astype(int))
    return {"auc": auc, "acc": acc, "f1": f1, "preds": preds, "labels": labels}


# ── Main ─────────────────────────────────────────────────────────────────────

def run(data_path: str = "data/raw/bbbp.csv",
        epochs: int = 120, batch_size: int = 16, lr: float = 1e-3, seed: int = 42):

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df     = pd.read_csv(data_path)
    graphs = [g for g in (smiles_to_graph(r.smiles, r.p_np)
                           for _, r in df.iterrows()) if g]
    print(f"Graphs: {len(graphs)}")

    train_g, test_g = train_test_split(
        graphs, test_size=0.2, random_state=seed,
        stratify=[g.y.item() for g in graphs])

    train_loader = DataLoader(train_g, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_g,  batch_size=batch_size)

    model     = BBBGraphNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    best_auc, history = 0, []
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, device)
        scheduler.step()
        history.append({"epoch": ep, "loss": loss, **{k: v for k, v in metrics.items()
                                                       if k not in ("preds","labels")}})
        if ep % 10 == 0:
            print(f"Epoch {ep:3d} | Loss {loss:.4f} | AUC {metrics['auc']:.4f} | Acc {metrics['acc']:.4f}")
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), "results/best_gnn.pt")

    print(f"\nBest Test AUC: {best_auc:.4f}")
    pd.DataFrame(history).to_csv("results/gnn_training_history.csv", index=False)
    return model, best_auc


if __name__ == "__main__":
    run()
