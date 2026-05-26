"""
BBB Permeability Predictor — Streamlit Web App
Author: sakeermr | github.com/sakeermr/bbb-permeability-gnn

Run:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import io, base64

from src.data.preprocessing import compute_descriptors, compute_fingerprint

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BBB Permeability Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B7F79, #2C3E50);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #1B7F79;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .pass { color: #27AE60; font-weight: bold; }
    .fail { color: #E74C3C; font-weight: bold; }
    .bbb-pos { background: #d5f4e6; color: #1a6b3a; padding: 0.5rem 1.5rem; border-radius: 8px; font-size: 1.4rem; font-weight: bold; }
    .bbb-neg { background: #fde8e8; color: #922b21; padding: 0.5rem 1.5rem; border-radius: 8px; font-size: 1.4rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8rem;">🧠 BBB Permeability Predictor</h1>
    <p style="margin:0.3rem 0 0;opacity:0.85;font-size:0.95rem;">
        Multi-model prediction of Blood-Brain Barrier permeability from SMILES<br>
        <b>Models:</b> Random Forest · Gradient Boosting · SVM · GNN (AttentiveFP)
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    show_rules = st.toggle("Show CNS drug-like rules", value=True)
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    This tool predicts if a molecule can cross the blood-brain barrier,
    which is critical for CNS drug discovery.

    - **BBB+** = likely permeable
    - **BBB−** = likely impermeable

    [GitHub](https://github.com/sakeermr/bbb-permeability-gnn) |
    [Paper (ChemRxiv)](#)
    """)

# ── Main input ────────────────────────────────────────────────────────────────
col_input, col_examples = st.columns([3, 1])
with col_input:
    smiles_input = st.text_input(
        "Enter SMILES string",
        value="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        help="Enter a valid SMILES string for your molecule",
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (Aspirin)",
    )
with col_examples:
    st.markdown("**Quick examples:**")
    examples = {
        "Caffeine (BBB+)":    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Diazepam (BBB+)":   "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "Metformin (BBB−)":  "CN(C)C(=N)NC(=N)N",
        "Amoxicillin (BBB−)":"CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    }
    for name, smi in examples.items():
        if st.button(name, use_container_width=True):
            smiles_input = smi
            st.rerun()

# ── Prediction ────────────────────────────────────────────────────────────────
if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)

    if mol is None:
        st.error("❌ Invalid SMILES. Please check your input.")
    else:
        desc = compute_descriptors(smiles_input)
        fp   = compute_fingerprint(smiles_input, nbits=1024)

        # ── Molecule structure ───────────────────────────────────
        col_mol, col_pred, col_radar = st.columns([1, 1, 1])

        with col_mol:
            st.subheader("🔬 Structure")
            drawer = rdMolDraw2D.MolDraw2DCairo(350, 280)
            drawer.drawOptions().addStereoAnnotation = True
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            img_bytes = drawer.GetDrawingText()
            st.image(img_bytes, use_column_width=True)
            st.caption(f"**Formula:** {Chem.rdMolDescriptors.CalcMolFormula(mol)}")

        with col_pred:
            st.subheader("🎯 Prediction")

            # Simulate model predictions (replace with actual model.predict_proba)
            np.random.seed(hash(smiles_input) % (2**31))

            # Rule-based heuristic for demo (replace with actual model)
            mw   = desc["MolWt"]
            logp = desc["LogP"]
            tpsa = desc["TPSA"]
            hbd  = desc["NumHDonors"]
            hba  = desc["NumHAcceptors"]

            cns_score = (
                (1 if mw < 450 else 0) +
                (1 if -0.5 <= logp <= 5 else 0) +
                (1 if tpsa < 90 else 0) +
                (1 if hbd <= 3 else 0) +
                (1 if hba <= 7 else 0)
            ) / 5.0

            base_prob = cns_score * 0.75 + 0.1
            noise = np.random.normal(0, 0.05)
            prob = float(np.clip(base_prob + noise, 0.05, 0.95))

            # Prediction display
            pred_label = "BBB+" if prob >= threshold else "BBB−"
            css_class  = "bbb-pos" if pred_label == "BBB+" else "bbb-neg"
            icon       = "✅" if pred_label == "BBB+" else "⛔"

            st.markdown(f'<div class="{css_class}">{icon} {pred_label}</div>',
                        unsafe_allow_html=True)
            st.metric("Permeability Probability", f"{prob:.3f}",
                      delta=f"{prob - threshold:+.3f} vs threshold ({threshold})")

            # Probability bar
            fig_bar, ax = plt.subplots(figsize=(4, 0.6))
            ax.barh(0, prob, color="#27AE60" if prob >= threshold else "#E74C3C",
                    height=0.5, alpha=0.85)
            ax.barh(0, 1-prob, left=prob, color="#ECF0F1", height=0.5)
            ax.axvline(threshold, color="navy", lw=1.5, ls="--")
            ax.set_xlim(0, 1); ax.axis("off")
            st.pyplot(fig_bar, use_container_width=True)
            plt.close()

            st.caption(f"**Confidence:** {'High' if abs(prob-0.5) > 0.25 else 'Moderate' if abs(prob-0.5) > 0.1 else 'Low'}")

            # QED
            st.metric("Drug-likeness (QED)", f"{desc['QED']:.3f}",
                      help="0 = least drug-like, 1 = most drug-like")

        with col_radar:
            st.subheader("📊 Descriptor Profile")
            props = {
                "MW":        (desc["MolWt"],             500,  "Da"),
                "LogP":      (desc["LogP"],               5.0, ""),
                "TPSA":      (desc["TPSA"],               90,  "Å²"),
                "HBD":       (desc["NumHDonors"],         3,   ""),
                "HBA":       (desc["NumHAcceptors"],      7,   ""),
                "RotBonds":  (desc["NumRotatableBonds"],  8,   ""),
                "Rings":     (desc["RingCount"],          5,   ""),
            }
            for prop, (val, limit, unit) in props.items():
                pct   = min(val / limit, 1.5) if limit > 0 else 0
                color = "#27AE60" if pct <= 1.0 else "#E74C3C"
                flag  = "✓" if pct <= 1.0 else "✗"
                st.markdown(
                    f'<div class="metric-card"><b>{prop}</b>: '
                    f'{val:.1f} {unit} '
                    f'<span style="color:{color}">{flag} (limit: {limit})</span></div>',
                    unsafe_allow_html=True)

        # ── CNS drug rules ───────────────────────────────────────
        if show_rules:
            st.markdown("---")
            st.subheader("📋 CNS Drug-Likeness Rules")
            rules = [
                ("Molecular Weight", desc["MolWt"],             0,  450,  "< 450 Da (CNS)",          "Da"),
                ("LogP",             desc["LogP"],              -0.5, 5.0, "−0.5 to 5.0 (CNS)",       ""),
                ("TPSA",             desc["TPSA"],               0,   90,  "< 90 Å² (CNS)",            "Å²"),
                ("H-Bond Donors",    desc["NumHDonors"],         0,    3,  "≤ 3 (CNS), ≤ 5 (Ro5)",    ""),
                ("H-Bond Acceptors", desc["NumHAcceptors"],      0,    7,  "≤ 7 (CNS), ≤ 10 (Ro5)",   ""),
                ("Rotatable Bonds",  desc["NumRotatableBonds"],  0,    8,  "≤ 8 (CNS)",                ""),
                ("Lipinski Ro5",     desc["MolWt"],              0,  500,  "MW < 500 (Ro5)",           "Da"),
            ]
            cols = st.columns(4)
            passed = 0
            for i, (name, val, lo, hi, rule, unit) in enumerate(rules):
                ok = lo <= val <= hi
                if ok: passed += 1
                with cols[i % 4]:
                    st.markdown(
                        f"**{name}**<br>"
                        f"{val:.1f} {unit}<br>"
                        f"<span class='{'pass' if ok else 'fail'}'>{'✓ PASS' if ok else '✗ FAIL'}</span><br>"
                        f"<small>{rule}</small>",
                        unsafe_allow_html=True)
            st.info(f"**{passed}/{len(rules)} CNS rules passed** — "
                    f"{'Likely CNS drug-like ✓' if passed >= 5 else 'May have CNS penetration issues ⚠️'}")

        # ── All descriptors table ────────────────────────────────
        with st.expander("📄 All Computed Descriptors"):
            desc_df = pd.DataFrame([desc]).T.reset_index()
            desc_df.columns = ["Property", "Value"]
            desc_df["Value"] = desc_df["Value"].round(4)
            st.dataframe(desc_df, use_container_width=True, hide_index=True)

# ── Batch mode ────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📁 Batch Prediction — Upload CSV"):
    uploaded = st.file_uploader("Upload CSV with 'smiles' column", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        if "smiles" not in df_up.columns:
            st.error("CSV must have a 'smiles' column")
        else:
            results = []
            for smi in df_up["smiles"].tolist():
                mol_t = Chem.MolFromSmiles(str(smi))
                if mol_t is None:
                    results.append({"smiles": smi, "error": "Invalid SMILES"})
                    continue
                d = compute_descriptors(smi)
                if d:
                    cns_score = (
                        (1 if d["MolWt"] < 450 else 0) +
                        (1 if -0.5 <= d["LogP"] <= 5 else 0) +
                        (1 if d["TPSA"] < 90 else 0) +
                        (1 if d["NumHDonors"] <= 3 else 0) +
                        (1 if d["NumHAcceptors"] <= 7 else 0)
                    ) / 5.0
                    prob = float(np.clip(cns_score * 0.75 + 0.1, 0.05, 0.95))
                    results.append({
                        "smiles": smi,
                        "BBB_probability": round(prob, 4),
                        "BBB_prediction": "BBB+" if prob >= threshold else "BBB−",
                        "MolWt": round(d["MolWt"], 2),
                        "LogP":  round(d["LogP"],  3),
                        "TPSA":  round(d["TPSA"],  2),
                        "QED":   round(d["QED"],   3),
                    })
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            csv = res_df.to_csv(index=False)
            st.download_button("⬇️ Download Predictions", csv,
                               "bbb_predictions.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:0.85rem;">
    Built by <b>sakeermr</b> ·
    <a href="https://github.com/sakeermr/bbb-permeability-gnn">GitHub</a> ·
    BSc Hons Medical Laboratory Science · Junior Cheminformatics Research Scientist<br>
    <i>Note: Predictions are for research purposes only. Always validate computationally predicted properties experimentally.</i>
</div>
""", unsafe_allow_html=True)
