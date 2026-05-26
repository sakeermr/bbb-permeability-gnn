"""
BBB Permeability Predictor — Models
Author: sakeermr

Baselines: Random Forest, XGBoost, SVM, Logistic Regression
Advanced:  GNN (AttentiveFP-style) architecture code for Colab/GPU execution
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def get_random_forest(n_estimators: int = 300, seed: int = 42) -> RandomForestClassifier:
    """Random Forest — strong baseline for molecular fingerprints."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def get_gradient_boosting(n_estimators: int = 200, seed: int = 42) -> GradientBoostingClassifier:
    """Gradient Boosting — often best tabular baseline."""
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )


def get_svm(C: float = 1.0, kernel: str = "rbf") -> Pipeline:
    """SVM with RBF kernel — classic cheminformatics baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=C, kernel=kernel, probability=True,
                    class_weight="balanced", random_state=42)),
    ])


def get_logistic_regression(C: float = 1.0) -> Pipeline:
    """Logistic Regression — interpretable baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000,
                                   class_weight="balanced", random_state=42)),
    ])


ALL_BASELINES = {
    "Random Forest":       get_random_forest,
    "Gradient Boosting":   get_gradient_boosting,
    "SVM (RBF)":           get_svm,
    "Logistic Regression": get_logistic_regression,
}


# ─────────────────────────────────────────────────────────────────────────────
# GNN ARCHITECTURE — Run this on Google Colab with GPU
# ─────────────────────────────────────────────────────────────────────────────
#
# Install in Colab:
#   !pip install torch torch-geometric rdkit-pypi
#
# This implements an AttentiveFP-style GNN (Xiong et al., JACS 2020)
# Molecule → Graph → Message Passing → Global Attention Readout → MLP → Label
# ─────────────────────────────────────────────────────────────────────────────

GNN_COLAB_CODE = '''
"""
AttentiveFP-style GNN for BBB Permeability Prediction
Run this on Google Colab (GPU recommended).
Author: sakeermr | github.com/sakeermr
"""

# ── Install ──────────────────────────────────────────────────────────────────
# !pip install torch torch-geometric rdkit-pypi scikit-learn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ── Atom & Bond Features ─────────────────────────────────────────────────────

ATOM_FEATURES = 9   # feature vector size per atom
BOND_FEATURES = 4   # feature vector size per bond

def atom_features(atom) -> list:
    """9-dimensional atom feature vector."""
    return [
        atom.GetAtomicNum() / 100.0,                         # atomic number (normalised)
        atom.GetDegree() / 10.0,                             # degree
        atom.GetFormalCharge() / 5.0,                        # formal charge
        int(atom.GetIsAromatic()),                           # aromaticity
        atom.GetTotalNumHs() / 8.0,                          # implicit Hs
        int(atom.IsInRing()),                                # in ring
        atom.GetMass() / 200.0,                              # mass (normalised)
        int(atom.GetHybridization() ==
            Chem.rdchem.HybridizationType.SP3),              # sp3
        int(atom.GetHybridization() ==
            Chem.rdchem.HybridizationType.SP2),              # sp2
    ]

def bond_features(bond) -> list:
    """4-dimensional bond feature vector."""
    bt = bond.GetBondTypeAsDouble()
    return [
        bt / 3.0,                          # bond type (normalised)
        int(bond.GetIsAromatic()),         # aromatic
        int(bond.IsInRing()),              # in ring
        int(bond.GetIsConjugated()),       # conjugated
    ]

def smiles_to_graph(smiles: str, label: int) -> Data | None:
    """Convert SMILES string to PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()],
                     dtype=torch.float)

    # Edge indices + edge attributes (bidirectional)
    edge_idx, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_idx += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if not edge_idx:  # single atom molecule
        edge_idx  = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURES), dtype=torch.float)
    else:
        edge_idx  = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, y=y,
                smiles=smiles)


# ── AttentiveFP-style GNN ───────────────────────────────────────────────────

class BBBGraphNet(nn.Module):
    """
    Graph Attention Network for BBB permeability.
    Architecture:
      Input → 2 GAT layers (message passing) →
      Global attention readout →
      3-layer MLP → sigmoid → BBB label
    """
    def __init__(self, in_channels: int = ATOM_FEATURES,
                 hidden: int = 64, heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        # GAT layers
        self.gat1 = GATConv(in_channels, hidden, heads=heads,
                            dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1,
                            dropout=dropout, concat=False)

        # Batch norm
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden)

        # Readout MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch

        # Message passing
        x = F.elu(self.bn1(self.gat1(x, ei)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.gat2(x, ei)))

        # Global readout (sum + mean pooling)
        x = global_add_pool(x, batch) + global_mean_pool(x, batch)

        return torch.sigmoid(self.mlp(x)).squeeze()


# ── Training loop ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).cpu().numpy()
        y    = batch.y.cpu().numpy()
        preds.extend(pred.flatten())
        labels.extend(y.flatten())
    auc = roc_auc_score(labels, preds)
    return auc, np.array(preds), np.array(labels)


# ── Main training script ─────────────────────────────────────────────────────

def run_gnn_training(data_path: str = "data/raw/bbbp.csv",
                     epochs: int = 100,
                     batch_size: int = 16,
                     lr: float = 1e-3,
                     seed: int = 42):

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(data_path)
    graphs = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row["smiles"], row["p_np"])
        if g:
            graphs.append(g)
    print(f"Loaded {len(graphs)} molecular graphs")

    # Split
    train_g, test_g = train_test_split(graphs, test_size=0.2,
                                        random_state=seed, stratify=[g.y.item() for g in graphs])
    train_loader = DataLoader(train_g, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_g,  batch_size=batch_size, shuffle=False)

    # Model
    model     = BBBGraphNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    # Train
    best_auc = 0
    history  = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        auc, _, _ = evaluate(model, test_loader, device)
        scheduler.step()
        history.append({"epoch": epoch, "loss": loss, "auc": auc})
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "results/best_gnn.pt")

    print(f"\\nBest AUC: {best_auc:.4f}")
    return model, best_auc, history


if __name__ == "__main__":
    run_gnn_training()
'''

# Save GNN code separately so it can be used in Colab
if __name__ == "__main__":
    with open("src/models/gnn_attentivefp.py", "w") as f:
        f.write(GNN_COLAB_CODE)
    print("GNN code exported.")
