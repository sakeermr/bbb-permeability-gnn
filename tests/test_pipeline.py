"""
BBB Permeability Predictor — Unit Tests
Author: sakeermr

Run:
    pytest tests/ -v
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocessing import (
    compute_descriptors, compute_fingerprint,
    get_scaffold, DESCRIPTOR_NAMES,
)
from src.evaluation.evaluate import compute_metrics


# ── Test molecules ────────────────────────────────────────────────────────────
VALID_SMILES   = "CC(=O)Oc1ccccc1C(=O)O"   # Aspirin
INVALID_SMILES = "INVALID_SMILES_XYZ"
CAFFEINE       = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
METFORMIN      = "CN(C)C(=N)NC(=N)N"


# ── Descriptor tests ──────────────────────────────────────────────────────────

class TestDescriptors:

    def test_valid_smiles_returns_dict(self):
        desc = compute_descriptors(VALID_SMILES)
        assert isinstance(desc, dict)

    def test_all_descriptor_keys_present(self):
        desc = compute_descriptors(VALID_SMILES)
        for key in DESCRIPTOR_NAMES:
            assert key in desc, f"Missing descriptor: {key}"

    def test_invalid_smiles_returns_none(self):
        desc = compute_descriptors(INVALID_SMILES)
        assert desc is None

    def test_aspirin_molwt(self):
        desc = compute_descriptors(VALID_SMILES)
        assert 170 < desc["MolWt"] < 195, f"Aspirin MW should be ~180, got {desc['MolWt']}"

    def test_aspirin_logp(self):
        desc = compute_descriptors(VALID_SMILES)
        assert -1.0 < desc["LogP"] < 3.5, f"Aspirin LogP out of range: {desc['LogP']}"

    def test_qed_range(self):
        desc = compute_descriptors(CAFFEINE)
        assert 0.0 <= desc["QED"] <= 1.0, f"QED must be 0–1, got {desc['QED']}"

    def test_descriptor_values_are_numeric(self):
        desc = compute_descriptors(VALID_SMILES)
        for key, val in desc.items():
            assert isinstance(val, (int, float)), f"{key} is not numeric: {type(val)}"

    def test_metformin_descriptors(self):
        desc = compute_descriptors(METFORMIN)
        assert desc is not None
        assert desc["MolWt"] < 200       # small molecule
        assert desc["NumHDonors"] >= 3   # multiple donors


# ── Fingerprint tests ─────────────────────────────────────────────────────────

class TestFingerprints:

    def test_fingerprint_shape_default(self):
        fp = compute_fingerprint(VALID_SMILES)
        assert fp is not None
        assert fp.shape == (2048,)

    def test_fingerprint_shape_custom(self):
        fp = compute_fingerprint(VALID_SMILES, nbits=1024)
        assert fp.shape == (1024,)

    def test_fingerprint_binary(self):
        fp = compute_fingerprint(VALID_SMILES)
        assert set(fp).issubset({0, 1}), "Fingerprint should be binary"

    def test_invalid_smiles_fingerprint_none(self):
        fp = compute_fingerprint(INVALID_SMILES)
        assert fp is None

    def test_different_molecules_different_fps(self):
        fp1 = compute_fingerprint(CAFFEINE)
        fp2 = compute_fingerprint(METFORMIN)
        assert not np.array_equal(fp1, fp2), "Different molecules should have different fingerprints"

    def test_same_smiles_same_fp(self):
        fp1 = compute_fingerprint(VALID_SMILES)
        fp2 = compute_fingerprint(VALID_SMILES)
        assert np.array_equal(fp1, fp2), "Same SMILES should give identical fingerprint"


# ── Scaffold tests ────────────────────────────────────────────────────────────

class TestScaffolds:

    def test_scaffold_returns_string(self):
        scaffold = get_scaffold(VALID_SMILES)
        assert isinstance(scaffold, str)

    def test_invalid_smiles_scaffold_none(self):
        scaffold = get_scaffold(INVALID_SMILES)
        assert scaffold is None

    def test_scaffold_is_valid_smiles(self):
        from rdkit import Chem
        scaffold = get_scaffold(CAFFEINE)
        mol = Chem.MolFromSmiles(scaffold)
        assert mol is not None, f"Scaffold is not valid SMILES: {scaffold}"


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:

    def test_perfect_classifier(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        m = compute_metrics(y_true, y_prob)
        assert m["AUC-ROC"] == 1.0
        assert m["Accuracy"] == 1.0

    def test_random_classifier(self):
        np.random.seed(42)
        y_true = np.array([1, 0] * 50)
        y_prob = np.random.uniform(0, 1, 100)
        m = compute_metrics(y_true, y_prob)
        assert 0.3 < m["AUC-ROC"] < 0.7, "Random classifier AUC should be near 0.5"

    def test_metrics_keys(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.8, 0.3, 0.7, 0.2])
        m = compute_metrics(y_true, y_prob)
        for key in ["AUC-ROC", "AUC-PR", "Accuracy", "F1", "Precision", "Recall"]:
            assert key in m, f"Missing metric: {key}"

    def test_all_metrics_in_range(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.1])
        m = compute_metrics(y_true, y_prob)
        for key, val in m.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0,1]"


# ── Integration test ──────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_single_molecule(self):
        """End-to-end: SMILES → descriptors + fingerprint → not None."""
        for smi in [VALID_SMILES, CAFFEINE, METFORMIN]:
            desc = compute_descriptors(smi)
            fp   = compute_fingerprint(smi, nbits=1024)
            assert desc is not None, f"Descriptors failed for {smi}"
            assert fp is not None,   f"Fingerprint failed for {smi}"
            assert len(fp) == 1024

    def test_lipinski_rule_of_five(self):
        """Known drug-like molecules should pass most Ro5 rules."""
        for smi in [VALID_SMILES, CAFFEINE]:
            desc = compute_descriptors(smi)
            assert desc["MolWt"] < 500,  f"MW > 500 for {smi}"
            assert desc["NumHDonors"] <= 5, f"HBD > 5 for {smi}"
            assert desc["NumHAcceptors"] <= 10, f"HBA > 10 for {smi}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
