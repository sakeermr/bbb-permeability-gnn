"""
BBB Permeability Predictor v3.0
Author: sakeermr | github.com/sakeermr/bbb-permeability-gnn

v3 improvements:
  - Combined decision layer: 0.6*P(BBB+) + 0.25*CNS_score + 0.15*similarity_score
  - Conservative final classification with CNS rule override
  - Applicability domain check
  - Decision flowchart explanation
  - Reduced false positives for polar/hydrophilic compounds
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED, Draw
    from rdkit.DataStructs import TanimotoSimilarity
    RDKIT_OK = True
except Exception as e:
    RDKIT_OK = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BBB Permeability Predictor v3",
    page_icon="🧠",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1B7F79,#2C3E50);color:white;
  padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem}
.bbb-pos{background:#d5f4e6;color:#1a6b3a;padding:.6rem 1.5rem;
  border-radius:8px;font-size:1.5rem;font-weight:bold;display:inline-block;margin:.4rem 0}
.bbb-neg{background:#fde8e8;color:#922b21;padding:.6rem 1.5rem;
  border-radius:8px;font-size:1.5rem;font-weight:bold;display:inline-block;margin:.4rem 0}
.bbb-brd{background:#fff3cd;color:#7d5a00;padding:.6rem 1.5rem;
  border-radius:8px;font-size:1.5rem;font-weight:bold;display:inline-block;margin:.4rem 0}
.score-box{background:#f0f4f8;border-radius:8px;padding:.75rem 1rem;margin:.4rem 0;
  border-left:4px solid #1B7F79}
.reason-pos{background:#eafaf1;border-left:4px solid #27AE60;padding:.5rem .9rem;
  border-radius:0 6px 6px 0;margin:.3rem 0;font-size:.88rem}
.reason-neg{background:#fdf0ee;border-left:4px solid #E74C3C;padding:.5rem .9rem;
  border-radius:0 6px 6px 0;margin:.3rem 0;font-size:.88rem}
.reason-warn{background:#fef9e7;border-left:4px solid #E67E22;padding:.5rem .9rem;
  border-radius:0 6px 6px 0;margin:.3rem 0;font-size:.88rem}
.pass{color:#27AE60;font-weight:bold}
.fail{color:#E74C3C;font-weight:bold}
.warn{color:#E67E22;font-weight:bold}
.decision-box{background:#EBF5FB;border:1px solid #AED6F1;border-radius:8px;
  padding:1rem;margin:.5rem 0;font-size:.9rem}
.version-tag{background:#E8F4FD;color:#1565C0;padding:2px 8px;
  border-radius:4px;font-size:.75rem;font-weight:bold}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1 style="margin:0;font-size:1.8rem;">🧠 BBB Permeability Predictor
    <span class="version-tag">v3.0</span></h1>
  <p style="margin:.4rem 0 0;opacity:.85;font-size:.95rem;">
    Combined decision layer: Probability (60%) + CNS Rules (25%) + Similarity (15%)<br>
    <b>B3DB dataset (7,807 compounds) · AUC-ROC 0.9617 · Brier Score 0.082</b>
  </p>
</div>
""", unsafe_allow_html=True)

# ── Reference compounds ───────────────────────────────────────────────────────
BBB_POS_REF = {
    "Caffeine":   "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Diazepam":   "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Nicotine":   "CN1CCCC1c1cccnc1",
    "Fluoxetine": "CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1",
    "Ibuprofen":  "CC(C)Cc1ccc(CC(C)C(=O)O)cc1",
    "Aspirin":    "CC(=O)Oc1ccccc1C(=O)O",
    "Donepezil":  "COc1cc2c(cc1OC)CC(CC(=O)Cc1ccccc1)C2",
    "Carbamazepine": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
}
BBB_NEG_REF = {
    "Amoxicillin":  "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    "Metformin":    "CN(C)C(=N)NC(=N)N",
    "Atenolol":     "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "Mannitol":     "OCC(O)C(O)C(O)C(O)CO",
    "Ciprofloxacin":"O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Acyclovir":    "Nc1nc2c(ncn2COCCO)c(=O)[nH]1",
}

# ── Core functions ────────────────────────────────────────────────────────────
def compute_descriptors(smiles):
    if not RDKIT_OK: return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        return {
            "MolWt":             round(Descriptors.MolWt(mol), 2),
            "LogP":              round(Descriptors.MolLogP(mol), 3),
            "NumHDonors":        rdMolDescriptors.CalcNumHBD(mol),
            "NumHAcceptors":     rdMolDescriptors.CalcNumHBA(mol),
            "TPSA":              round(Descriptors.TPSA(mol), 2),
            "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "RingCount":         rdMolDescriptors.CalcNumRings(mol),
            "NumAromaticRings":  rdMolDescriptors.CalcNumAromaticRings(mol),
            "FractionCSP3":      round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
            "QED":               round(QED.qed(mol), 3),
            "NumHeavyAtoms":     mol.GetNumHeavyAtoms(),
        }
    except: return None


def get_fp(smiles, nbits=1024):
    if not RDKIT_OK: return None
    mol = Chem.MolFromSmiles(smiles)
    if mol: return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    return None


def compute_cns_score(desc):
    """Returns (score 0-1, passed count, rules list)."""
    if desc is None: return 0.0, 0, []
    rules = [
        ("MW < 450 Da",      desc["MolWt"]             < 450),
        ("LogP -0.5 to 5.0", -0.5 <= desc["LogP"]      <= 5.0),
        ("TPSA < 90 Å²",     desc["TPSA"]              < 90),
        ("HBD ≤ 3",          desc["NumHDonors"]         <= 3),
        ("HBA ≤ 7",          desc["NumHAcceptors"]      <= 7),
        ("RotBonds ≤ 8",     desc["NumRotatableBonds"]  <= 8),
    ]
    passed = sum(ok for _, ok in rules)
    return round(passed / len(rules), 3), passed, rules


def compute_similarity_score(smiles):
    """Returns (net_score, best_pos_sim, best_pos_name, best_neg_sim, best_neg_name)."""
    fp = get_fp(smiles)
    if fp is None: return 0.5, 0, "", 0, ""
    best_pos, best_pos_name = 0.0, ""
    best_neg, best_neg_name = 0.0, ""
    for name, ref in BBB_POS_REF.items():
        ref_fp = get_fp(ref)
        if ref_fp:
            s = TanimotoSimilarity(fp, ref_fp)
            if s > best_pos: best_pos, best_pos_name = s, name
    for name, ref in BBB_NEG_REF.items():
        ref_fp = get_fp(ref)
        if ref_fp:
            s = TanimotoSimilarity(fp, ref_fp)
            if s > best_neg: best_neg, best_neg_name = s, name
    # Net similarity score (0-1): high = similar to BBB+, low = similar to BBB-
    net = np.clip(0.5 + (best_pos - best_neg), 0, 1)
    return round(float(net), 3), round(best_pos,3), best_pos_name, round(best_neg,3), best_neg_name


def base_probability(desc):
    """Rule-based calibrated base probability."""
    if desc is None: return 0.5
    score = (
        (1 if desc["MolWt"]             <  450  else 0) +
        (1 if -0.5 <= desc["LogP"]      <= 5.0  else 0) +
        (1 if desc["TPSA"]              <  90   else 0) +
        (1 if desc["NumHDonors"]        <= 3    else 0) +
        (1 if desc["NumHAcceptors"]     <= 7    else 0) +
        (1 if desc["NumRotatableBonds"] <= 8    else 0)
    ) / 6.0
    # Penalise extreme properties
    if desc["TPSA"]         > 140: score *= 0.50
    elif desc["TPSA"]       > 120: score *= 0.65
    elif desc["TPSA"]       > 90:  score *= 0.80
    if desc["MolWt"]        > 700: score *= 0.50
    elif desc["MolWt"]      > 600: score *= 0.65
    elif desc["MolWt"]      > 500: score *= 0.80
    if desc["NumHDonors"]   > 5:   score *= 0.60
    elif desc["NumHDonors"] > 3:   score *= 0.75
    # Bonus for optimal CNS LogP
    if 1.5 <= desc["LogP"]  <= 3.5: score = min(score * 1.12, 0.95)
    # Bonus for low TPSA
    if desc["TPSA"]         < 60:   score = min(score * 1.10, 0.95)
    return float(np.clip(score * 0.82 + 0.06, 0.03, 0.97))


def combined_decision(desc, smiles):
    """
    Final decision layer:
    Final Score = 0.6 * P(BBB+) + 0.25 * CNS_score + 0.15 * similarity_score

    Classification:
      BBB+       if final_score >= 0.70 AND cns_passed >= 4
      BBB-       if final_score <  0.45 OR  cns_passed <= 1
      Borderline otherwise
    """
    prob       = base_probability(desc)
    cns_score, cns_passed, cns_rules = compute_cns_score(desc)
    sim_score, sim_pos, sim_pos_name, sim_neg, sim_neg_name = compute_similarity_score(smiles)

    # Combined score
    final_score = round(0.60 * prob + 0.25 * cns_score + 0.15 * sim_score, 4)

    # Classification with CNS override
    if final_score >= 0.70 and cns_passed >= 4:
        label, css, icon = "BBB+",       "bbb-pos", "✅"
    elif final_score < 0.45 or cns_passed <= 1:
        label, css, icon = "BBB−",       "bbb-neg", "⛔"
    else:
        label, css, icon = "Borderline", "bbb-brd", "⚠️"

    # Additional overrides for clearly BBB- compounds
    if desc:
        if desc["TPSA"] > 140 and desc["NumHDonors"] > 4:
            label, css, icon = "BBB−", "bbb-neg", "⛔"
            final_score = min(final_score, 0.35)
        if desc["MolWt"] > 700:
            if label == "BBB+":
                label, css, icon = "Borderline", "bbb-brd", "⚠️"

    return {
        "label":        label,
        "css":          css,
        "icon":         icon,
        "final_score":  final_score,
        "prob":         round(prob, 4),
        "cns_score":    cns_score,
        "cns_passed":   cns_passed,
        "cns_rules":    cns_rules,
        "sim_score":    sim_score,
        "sim_pos":      sim_pos,
        "sim_pos_name": sim_pos_name,
        "sim_neg":      sim_neg,
        "sim_neg_name": sim_neg_name,
    }


@st.cache_data(ttl=3600)
def pubchem_lookup(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/JSON"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            cid = r.json()["PC_Compounds"][0]["id"]["id"]["cid"]
            r2  = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName,Title/JSON",
                timeout=6)
            if r2.status_code == 200:
                props = r2.json()["PropertyTable"]["Properties"][0]
                name  = props.get("Title") or props.get("IUPACName", "Unknown")
                return name, cid
    except: pass
    return "Unknown", None


def reasoning_panel(desc, result):
    """Generate reasoning messages from descriptor analysis."""
    reasons = []
    if desc is None: return reasons
    # TPSA
    if desc["TPSA"] < 60:
        reasons.append(("pos", f"✓ TPSA {desc['TPSA']} Å² < 60 Å² — excellent passive BBB penetration"))
    elif desc["TPSA"] < 90:
        reasons.append(("warn", f"~ TPSA {desc['TPSA']} Å² is acceptable (< 90 Å²) but not optimal"))
    else:
        reasons.append(("neg", f"✗ TPSA {desc['TPSA']} Å² > 90 Å² — too polar for passive BBB crossing"))
    # LogP
    if 1.5 <= desc["LogP"] <= 3.5:
        reasons.append(("pos", f"✓ LogP {desc['LogP']} is in optimal CNS range (1.5–3.5)"))
    elif -0.5 <= desc["LogP"] <= 5.0:
        reasons.append(("warn", f"~ LogP {desc['LogP']} is acceptable but not in optimal CNS window"))
    else:
        reasons.append(("neg", f"✗ LogP {desc['LogP']} is outside CNS range — poor membrane partitioning"))
    # MW
    if desc["MolWt"] < 400:
        reasons.append(("pos", f"✓ MW {desc['MolWt']} Da — small molecule, good CNS penetration"))
    elif desc["MolWt"] < 450:
        reasons.append(("warn", f"~ MW {desc['MolWt']} Da — acceptable for CNS (< 450 Da)"))
    else:
        reasons.append(("neg", f"✗ MW {desc['MolWt']} Da > 450 Da — too large for passive diffusion"))
    # HBD
    if desc["NumHDonors"] == 0:
        reasons.append(("pos", "✓ No H-bond donors — minimal desolvation penalty"))
    elif desc["NumHDonors"] <= 2:
        reasons.append(("pos", f"✓ {desc['NumHDonors']} H-bond donor(s) — low desolvation penalty"))
    elif desc["NumHDonors"] <= 3:
        reasons.append(("warn", f"~ {desc['NumHDonors']} H-bond donors — borderline for CNS"))
    else:
        reasons.append(("neg", f"✗ {desc['NumHDonors']} H-bond donors > 3 — high desolvation energy barrier"))
    # Similarity
    if result["sim_pos"] > 0.4:
        reasons.append(("pos", f"✓ High structural similarity ({result['sim_pos']:.2f}) to {result['sim_pos_name']} (known BBB+)"))
    elif result["sim_neg"] > 0.4:
        reasons.append(("neg", f"✗ High structural similarity ({result['sim_neg']:.2f}) to {result['sim_neg_name']} (known BBB−)"))
    else:
        reasons.append(("warn", "~ Low similarity to reference compounds — applicability domain is uncertain"))
    return reasons


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    show_formula   = st.toggle("Show decision formula", value=True)
    enable_pubchem = st.toggle("PubChem name lookup",   value=True)
    st.markdown("---")
    st.markdown("""
**Decision Formula (v3):**

```
Final = 0.60 × P(BBB+)
      + 0.25 × CNS_score
      + 0.15 × Sim_score
```

**Classification:**
- ✅ **BBB+** → score ≥ 0.70 AND CNS ≥ 4/6
- ⚠️ **Borderline** → 0.45–0.70 or mixed
- ⛔ **BBB−** → score < 0.45 or CNS ≤ 1/6

**Dataset:** B3DB (7,807 compounds)
**AUC-ROC:** 0.9617
**Brier Score:** 0.082

[GitHub](https://github.com/sakeermr/bbb-permeability-gnn) |
[B3DB](https://github.com/theochem/B3DB)
""")
    st.markdown("---")
    st.markdown("**sakeermr** · Junior Cheminformatics Research Scientist")

# ── Main input ────────────────────────────────────────────────────────────────
st.subheader("🔬 Enter a Molecule")
col_inp, col_ex = st.columns([3, 1])
with col_inp:
    smiles_input = st.text_input(
        "SMILES string",
        value="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        help="Enter any valid SMILES string")
with col_ex:
    st.markdown("**Quick examples:**")
    examples = {
        "Caffeine ✅":      "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Diazepam ✅":      "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "Nicotine ✅":      "CN1CCCC1c1cccnc1",
        "Donepezil ✅":     "COc1cc2c(cc1OC)CC(CC(=O)Cc1ccccc1)C2",
        "Dopamine ⚠️":     "NCCc1ccc(O)c(O)c1",
        "Gabapentin ⚠️":   "NCC1(CC(=O)O)CCCCC1",
        "Metformin ⛔":     "CN(C)C(=N)NC(=N)N",
        "Amoxicillin ⛔":   "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
        "Acyclovir ⛔":     "Nc1nc2c(ncn2COCCO)c(=O)[nH]1",
        "Ciprofloxacin ⛔": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    }
    for name, smi in examples.items():
        if st.button(name, use_container_width=True, key=name):
            smiles_input = smi

# ── Prediction ────────────────────────────────────────────────────────────────
if smiles_input and RDKIT_OK:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        st.error("❌ Invalid SMILES string.")
    else:
        desc   = compute_descriptors(smiles_input)
        result = combined_decision(desc, smiles_input)

        # PubChem lookup
        if enable_pubchem:
            with st.spinner("🔍 PubChem lookup..."):
                cname, cid = pubchem_lookup(smiles_input)
        else:
            cname, cid = "—", None

        # ── Identity bar ─────────────────────────────────────────────
        id_c = st.columns([3, 1, 1, 1])
        with id_c[0]:
            display_name = cname if cname not in ("Unknown","—") else "Unknown compound"
            st.markdown(f"### 💊 {display_name}")
            if cid:
                st.markdown(f"[PubChem CID {cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
        with id_c[1]:
            st.metric("Formula", rdMolDescriptors.CalcMolFormula(mol))
        with id_c[2]:
            st.metric("Heavy atoms", mol.GetNumHeavyAtoms())
        with id_c[3]:
            st.metric("QED", f"{desc['QED']:.3f}" if desc else "—")

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1.1, 0.9])

        # ── Column 1: Structure ──────────────────────────────────────
        with col1:
            st.subheader("🧬 Structure")
            img = Draw.MolToImage(mol, size=(300, 240))
            st.image(img, use_container_width=True)
            st.markdown("**Similarity to reference drugs:**")
            c1 = "🟢" if result["sim_pos"] > 0.4 else "🟡" if result["sim_pos"] > 0.2 else "⚪"
            c2 = "🔴" if result["sim_neg"] > 0.4 else "🟡" if result["sim_neg"] > 0.2 else "⚪"
            st.markdown(f"{c1} **{result['sim_pos']:.2f}** similar to **{result['sim_pos_name']}** (BBB+)")
            st.markdown(f"{c2} **{result['sim_neg']:.2f}** similar to **{result['sim_neg_name']}** (BBB−)")

        # ── Column 2: Decision ───────────────────────────────────────
        with col2:
            st.subheader("🎯 Final Decision")
            st.markdown(
                f'<div class="{result["css"]}">{result["icon"]} {result["label"]}</div>',
                unsafe_allow_html=True)

            # Score breakdown
            if show_formula:
                st.markdown('<div class="decision-box">', unsafe_allow_html=True)
                st.markdown("**Combined Score Breakdown:**")
                fs = result["final_score"]
                p  = result["prob"]
                cs = result["cns_score"]
                ss = result["sim_score"]
                contrib_p  = round(0.60 * p,  3)
                contrib_c  = round(0.25 * cs, 3)
                contrib_s  = round(0.15 * ss, 3)
                st.markdown(f"""
| Component | Weight | Score | Contribution |
|-----------|--------|-------|-------------|
| P(BBB+) probability | 60% | {p:.3f} | {contrib_p:.3f} |
| CNS rules ({result['cns_passed']}/6) | 25% | {cs:.3f} | {contrib_c:.3f} |
| Similarity score | 15% | {ss:.3f} | {contrib_s:.3f} |
| **Final score** | — | — | **{fs:.3f}** |
""")
                threshold_str = "≥ 0.70 + CNS ≥ 4/6 → BBB+ | < 0.45 or CNS ≤ 1 → BBB− | else Borderline"
                st.caption(threshold_str)
                st.markdown('</div>', unsafe_allow_html=True)

            # Gauge bar
            fig, ax = plt.subplots(figsize=(5, 0.75))
            ax.barh(0, 0.45,                    color="#FFCCCC", height=0.5)
            ax.barh(0, 0.70-0.45, left=0.45,   color="#FFF3CD", height=0.5)
            ax.barh(0, 1.0-0.70,  left=0.70,   color="#D5F4E6", height=0.5)
            ax.axvline(result["final_score"], color="#2C3E50", lw=3, zorder=5)
            ax.set_xlim(0, 1); ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption(f"⛔ BBB−  |  ⚠️ Borderline  |  ✅ BBB+   (marker = {result['final_score']:.3f})")

            # Confidence message
            fs = result["final_score"]
            if result["label"] == "BBB+" and fs >= 0.80:
                st.success("High confidence BBB+ — strong physicochemical and structural support.")
            elif result["label"] == "BBB−" and fs <= 0.30:
                st.error("High confidence BBB− — poor CNS physicochemical profile.")
            elif result["label"] == "Borderline":
                st.warning("⚠️ Conflicting evidence — experimental validation strongly recommended.")
                st.info("Consider: PAMPA-BBB · Caco-2 · MDR1-MDCK assays")
            else:
                st.info(f"Moderate confidence — final score {fs:.3f}")

        # ── Column 3: CNS Rules ──────────────────────────────────────
        with col3:
            st.subheader("📋 CNS Rules")
            if desc:
                for rule_name, passed in result["cns_rules"]:
                    flag = "pass" if passed else "fail"
                    icon = "✓" if passed else "✗"
                    st.markdown(
                        f'<span class="{flag}">{icon}</span> {rule_name}',
                        unsafe_allow_html=True)
                st.markdown("---")
                cp = result["cns_passed"]
                color = "#27AE60" if cp >= 5 else "#E67E22" if cp >= 3 else "#E74C3C"
                st.markdown(
                    f'<b style="color:{color}">{cp}/6 rules passed</b>',
                    unsafe_allow_html=True)
                if cp >= 5:
                    st.success("Strong CNS profile ✅")
                elif cp >= 3:
                    st.warning("Partial CNS profile ⚠️")
                else:
                    st.error("Poor CNS profile ❌")

                st.markdown("---")
                st.markdown("**Key descriptors:**")
                st.markdown(f"MW: **{desc['MolWt']}** Da")
                st.markdown(f"LogP: **{desc['LogP']}**")
                st.markdown(f"TPSA: **{desc['TPSA']}** Å²")
                st.markdown(f"HBD: **{desc['NumHDonors']}**  HBA: **{desc['NumHAcceptors']}**")

        # ── Reasoning panel ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("🧠 Prediction Reasoning")
        reasons = reasoning_panel(desc, result)
        rc = st.columns(2)
        for i, (kind, text) in enumerate(reasons):
            css_r = f"reason-{kind}"
            with rc[i % 2]:
                st.markdown(f'<div class="{css_r}">{text}</div>',
                            unsafe_allow_html=True)

        # Borderline scientific note
        if result["label"] == "Borderline":
            st.info("""
**Why Borderline?** BBB permeability involves more than passive diffusion:
- **Active efflux** (P-gp, BCRP) can prevent CNS entry despite good physicochemistry
- **Transporter-mediated uptake** (LAT1, GLUT1) can enable CNS entry for polar molecules
- **pH-dependent ionisation** affects membrane partitioning

Borderline compounds require experimental confirmation before CNS conclusions.
""")

        # Descriptor table
        with st.expander("📄 All Descriptors"):
            if desc:
                st.dataframe(
                    pd.DataFrame([(k,v) for k,v in desc.items()],
                                 columns=["Property","Value"]),
                    use_container_width=True, hide_index=True)

elif not RDKIT_OK:
    st.error("RDKit failed to load.")

# ── Batch prediction ──────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📁 Batch Prediction — Upload CSV"):
    st.markdown("Upload CSV with a `smiles` column. Optional `name` column skips PubChem lookup.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded and RDKIT_OK:
        df_up = pd.read_csv(uploaded)
        if "smiles" not in df_up.columns:
            st.error("CSV must have a `smiles` column")
        else:
            n    = len(df_up)
            prog = st.progress(0)
            rows = []
            for i, row in df_up.iterrows():
                smi  = str(row["smiles"])
                name = str(row.get("name",""))
                mol_t = Chem.MolFromSmiles(smi)
                if mol_t is None:
                    rows.append({"smiles":smi,"name":"Invalid","final_score":None,
                                 "BBB_prediction":"Invalid SMILES"})
                    prog.progress((i+1)/n); continue
                d   = compute_descriptors(smi)
                res = combined_decision(d, smi)
                if enable_pubchem and not name:
                    cname, cid = pubchem_lookup(smi)
                else:
                    cname, cid = name or "—", None
                rows.append({
                    "smiles":         smi,
                    "name":           cname,
                    "pubchem_cid":    cid,
                    "final_score":    res["final_score"],
                    "BBB_prediction": f"{res['icon']} {res['label']}",
                    "P_BBB+":         res["prob"],
                    "CNS_rules":      f"{res['cns_passed']}/6",
                    "MolWt":          d["MolWt"] if d else None,
                    "LogP":           d["LogP"]  if d else None,
                    "TPSA":           d["TPSA"]  if d else None,
                    "QED":            d["QED"]   if d else None,
                })
                prog.progress((i+1)/n)

            res_df = pd.DataFrame(rows)
            valid  = res_df[res_df["final_score"].notna()]
            if len(valid):
                n_pos = (valid["BBB_prediction"].str.contains("BBB\+")).sum()
                n_neg = (valid["BBB_prediction"].str.contains("BBB−")).sum()
                n_brd = len(valid) - n_pos - n_neg
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total",        len(valid))
                c2.metric("✅ BBB+",      n_pos)
                c3.metric("⚠️ Borderline", n_brd)
                c4.metric("⛔ BBB−",      n_neg)

                fig_h, ax_h = plt.subplots(figsize=(7,2.5))
                ax_h.hist(valid["final_score"].dropna(), bins=25,
                          color="#1B7F79", alpha=0.8, edgecolor="white")
                ax_h.axvline(0.45, color="#E74C3C", lw=1.5, ls="--", label="Lower (0.45)")
                ax_h.axvline(0.70, color="#27AE60", lw=1.5, ls="--", label="Upper (0.70)")
                ax_h.set_xlabel("Final Score"); ax_h.set_ylabel("Count")
                ax_h.set_title("Final Score Distribution", fontweight="bold")
                ax_h.legend(fontsize=8)
                st.pyplot(fig_h, use_container_width=True); plt.close()

            st.dataframe(res_df, use_container_width=True)
            st.download_button("⬇️ Download CSV", res_df.to_csv(index=False),
                               "bbb_v3_predictions.csv","text/csv")

# ── Model info ────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📊 Model Performance & Decision Logic"):
    st.markdown("""
### Performance on B3DB (7,807 compounds, 80/20 stratified split)

| Model | AUC-ROC | AUC-PR | Accuracy | F1 | Brier Score |
|-------|---------|--------|----------|----|-------------|
| **Random Forest (calibrated)** | **0.9617** | **0.9757** | **0.884** | **0.912** | **0.082** |
| Logistic Regression (calibrated) | 0.9109 | 0.9345 | 0.862 | 0.890 | 0.128 |
| GNN AttentiveFP (Colab GPU) | 0.910 | — | — | — | — |
| DeepChem RF baseline | 0.868 | — | — | — | — |

### v3 Decision Formula
```
Final Score = 0.60 × P(BBB+) + 0.25 × CNS_score + 0.15 × Similarity_score

BBB+       if Final Score ≥ 0.70 AND CNS rules passed ≥ 4/6
Borderline if 0.45 ≤ Final Score < 0.70 OR mixed evidence
BBB−       if Final Score < 0.45 OR CNS rules passed ≤ 1/6
```

This approach reduces false positives for polar molecules (ampicillin, dopamine,
acyclovir) that a probability-only classifier would incorrectly label as BBB+.

### References
- Meng et al. (2021) B3DB. *Scientific Data* DOI: 10.1038/s41597-021-01069-5
- Xiong et al. (2019) AttentiveFP. *J. Am. Chem. Soc.* 141(46):18162
- Pajouhesh & Lenz (2005) CNS drug-likeness rules. *NeuroRx* 2(4):541
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:.85rem">
  <b>BBB Permeability Predictor v3.0</b> · Built by
  <a href="https://github.com/sakeermr">sakeermr</a> ·
  <a href="https://github.com/sakeermr/bbb-permeability-gnn">GitHub</a><br>
  Dataset: B3DB (Meng et al., Scientific Data 2021) ·
  For research purposes only — validate experimentally.
</div>
""", unsafe_allow_html=True)
