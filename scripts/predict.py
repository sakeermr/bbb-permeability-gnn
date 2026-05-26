"""
BBB Permeability Predictor — Command-line prediction
Author: sakeermr | github.com/sakeermr/bbb-permeability-gnn

Usage:
    # Single SMILES
    python scripts/predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"

    # Multiple SMILES from CSV
    python scripts/predict.py --csv my_molecules.csv --smiles_col smiles

    # Save predictions
    python scripts/predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --output results/predictions.csv
"""

import argparse
import sys
import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import compute_descriptors, compute_fingerprint, DESCRIPTOR_NAMES

# ── Interpretability thresholds (from Lipinski & CNS drug rules) ────────────
CNS_RULES = {
    "MolWt":             (0, 450,    "MW < 450 for CNS drugs"),
    "LogP":              (−0.5, 5.0, "LogP −0.5 to 5.0"),
    "NumHDonors":        (0, 3,      "HBD ≤ 3"),
    "NumHAcceptors":     (0, 7,      "HBA ≤ 7"),
    "TPSA":              (0, 90,     "TPSA < 90 Å²"),
    "NumRotatableBonds": (0, 8,      "RotBonds ≤ 8"),
}


def check_cns_rules(desc: dict) -> dict:
    """Check CNS drug-likeness rules."""
    results = {}
    for prop, (lo, hi, rule) in CNS_RULES.items():
        val = desc.get(prop, None)
        if val is not None:
            results[prop] = {
                "value": round(val, 3),
                "pass": lo <= val <= hi,
                "rule": rule,
            }
    return results


def predict_single(smiles: str, model=None, scaler=None) -> dict:
    """Predict BBB permeability for a single SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "error": "Invalid SMILES"}

    desc = compute_descriptors(smiles)
    fp   = compute_fingerprint(smiles, nbits=1024)

    if desc is None or fp is None:
        return {"smiles": smiles, "error": "Feature computation failed"}

    result = {
        "smiles":    smiles,
        "MolWt":     round(desc["MolWt"], 2),
        "LogP":      round(desc["LogP"],  3),
        "TPSA":      round(desc["TPSA"],  2),
        "QED":       round(desc["QED"],   3),
        "HBD":       desc["NumHDonors"],
        "HBA":       desc["NumHAcceptors"],
        "RotBonds":  desc["NumRotatableBonds"],
    }

    if model is not None:
        X = fp.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        result["BBB_probability"] = round(float(prob), 4)
        result["BBB_prediction"]  = "BBB+" if prob >= 0.5 else "BBB−"
        result["confidence"]      = "High" if abs(prob - 0.5) > 0.25 else "Moderate" if abs(prob - 0.5) > 0.1 else "Low"

    cns = check_cns_rules(desc)
    passed = sum(v["pass"] for v in cns.values())
    result["CNS_rules_passed"] = f"{passed}/{len(cns)}"
    result["CNS_drug_like"]    = passed >= 5

    return result


def batch_predict(smiles_list: list, model=None, scaler=None) -> pd.DataFrame:
    """Predict for a list of SMILES strings."""
    results = [predict_single(s, model, scaler) for s in smiles_list]
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Predict BBB permeability from SMILES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--smiles",     type=str, help="Single SMILES string")
    parser.add_argument("--csv",        type=str, help="CSV file with SMILES")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="Column name for SMILES")
    parser.add_argument("--model",      type=str, default="results/best_rf_model.pkl", help="Path to model .pkl")
    parser.add_argument("--output",     type=str, help="Output CSV path")
    parser.add_argument("--no_model",   action="store_true", help="Skip ML prediction, show descriptors only")

    args = parser.parse_args()

    # Load model if available
    model, scaler = None, None
    if not args.no_model and os.path.exists(args.model):
        import pickle
        with open(args.model, "rb") as f:
            saved = pickle.load(f)
            model  = saved.get("model")
            scaler = saved.get("scaler")
        print(f"Loaded model from {args.model}")
    else:
        print("No model loaded — showing descriptors only. Run scripts/train_baselines.py first.")

    # Predict
    if args.smiles:
        result = predict_single(args.smiles, model, scaler)
        print("\n" + "="*50)
        print("BBB Permeability Prediction")
        print("="*50)
        for k, v in result.items():
            print(f"  {k:<22}: {v}")
        print("="*50)

        if args.output:
            pd.DataFrame([result]).to_csv(args.output, index=False)
            print(f"\nSaved → {args.output}")

    elif args.csv:
        df = pd.read_csv(args.csv)
        smiles_list = df[args.smiles_col].tolist()
        print(f"Processing {len(smiles_list)} molecules...")
        results_df = batch_predict(smiles_list, model, scaler)
        print(results_df.to_string(index=False))

        out = args.output or "results/batch_predictions.csv"
        results_df.to_csv(out, index=False)
        print(f"\nSaved → {out}")

    else:
        parser.print_help()
        # Demo with known drugs
        print("\n\nDemo — predicting for 5 known drugs:")
        demo = [
            ("Aspirin (BBB+)",      "CC(=O)Oc1ccccc1C(=O)O"),
            ("Caffeine (BBB+)",     "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
            ("Diazepam (BBB+)",     "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc21"),
            ("Metformin (BBB−)",    "CN(C)C(=N)NC(=N)N"),
            ("Amoxicillin (BBB−)",  "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O"),
        ]
        for name, smi in demo:
            r = predict_single(smi, model, scaler)
            prob = r.get("BBB_probability", "N/A")
            pred = r.get("BBB_prediction", "N/A")
            print(f"  {name:<30} → {pred}  (prob={prob})")


if __name__ == "__main__":
    main()
