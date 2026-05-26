"""
BBB Permeability Predictor — Data Preprocessing
Author: sakeermr
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ── Lipinski + extended descriptors ─────────────────────────────────────────

DESCRIPTOR_NAMES = [
    "MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA",
    "NumRotatableBonds", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticRings", "RingCount", "FractionCSP3", "QED",
    "MolMR", "NumHeavyAtoms", "NumHeteroatoms",
]

def compute_descriptors(smiles: str) -> dict | None:
    """Compute physicochemical descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt":             Descriptors.MolWt(mol),
        "LogP":              Descriptors.MolLogP(mol),
        "NumHDonors":        rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors":     rdMolDescriptors.CalcNumHBA(mol),
        "TPSA":              Descriptors.TPSA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "NumAromaticRings":  rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "RingCount":         rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3":      rdMolDescriptors.CalcFractionCSP3(mol),
        "QED":               QED.qed(mol),
        "MolMR":             Descriptors.MolMR(mol),
        "NumHeavyAtoms":     mol.GetNumHeavyAtoms(),
        "NumHeteroatoms":    rdMolDescriptors.CalcNumHeteroatoms(mol),
    }


def compute_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048) -> np.ndarray | None:
    """Morgan (ECFP4) fingerprint as numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp)


def get_scaffold(smiles: str) -> str | None:
    """Return Bemis-Murcko scaffold SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


# ── Dataset loading & cleaning ───────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """Load BBBP CSV, validate SMILES, add descriptors + fingerprints."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")

    records = []
    for _, row in df.iterrows():
        smi = row["smiles"]
        label = row["p_np"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        desc = compute_descriptors(smi)
        if desc is None:
            continue
        fp = compute_fingerprint(smi)
        if fp is None:
            continue
        scaffold = get_scaffold(smi)
        record = {"name": row.get("name", ""), "smiles": smi,
                  "p_np": int(label), "scaffold": scaffold, **desc}
        records.append(record)

    clean_df = pd.DataFrame(records)
    print(f"After cleaning: {len(clean_df)} valid compounds")
    print(f"Class balance — BBB+: {clean_df['p_np'].sum()}  BBB-: {(clean_df['p_np']==0).sum()}")
    return clean_df


# ── Scaffold-based train/test split ─────────────────────────────────────────

def scaffold_split(df: pd.DataFrame, test_size: float = 0.2,
                   seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scaffold split: groups compounds by Murcko scaffold,
    puts whole scaffold groups into train or test.
    Prevents data leakage from structurally similar molecules.
    """
    np.random.seed(seed)
    scaffolds = {}
    for idx, row in df.iterrows():
        s = row["scaffold"] or row["smiles"]
        scaffolds.setdefault(s, []).append(idx)

    scaffold_groups = list(scaffolds.values())
    np.random.shuffle(scaffold_groups)

    n_test = int(len(df) * test_size)
    test_idx, train_idx = [], []
    for group in scaffold_groups:
        if len(test_idx) < n_test:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    return train_df, test_df


# ── Feature matrix builders ──────────────────────────────────────────────────

def get_descriptor_matrix(df: pd.DataFrame,
                           scaler: StandardScaler | None = None,
                           fit: bool = True):
    """Return scaled descriptor matrix + fitted scaler."""
    X = df[DESCRIPTOR_NAMES].values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler


def get_fingerprint_matrix(df: pd.DataFrame,
                            radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """Return Morgan fingerprint matrix."""
    fps = [compute_fingerprint(s, radius, nbits) for s in df["smiles"]]
    return np.array(fps, dtype=np.float32)


def get_combined_features(df: pd.DataFrame,
                           scaler: StandardScaler | None = None,
                           fit: bool = True,
                           nbits: int = 1024):
    """Descriptors + fingerprints concatenated."""
    desc, scaler = get_descriptor_matrix(df, scaler, fit)
    fps = get_fingerprint_matrix(df, nbits=nbits)
    return np.hstack([desc, fps]), scaler
