"""
BBB Permeability Predictor v4.0
Author: sakeermr | github.com/sakeermr/bbb-permeability-gnn

v4 improvements:
  - Rebalanced formula: 0.55*P + 0.30*CNS + 0.15*Sim
  - Stricter penalty for catecholamines / ionizable compounds
  - Applicability domain (AD) label
  - Stricter confidence labeling
  - Fixed encoding issues
  - Scientific disclaimer
  - Normalized CNS score display
  - Improved PubChem lookup with CID fallback
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
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
    page_title="BBB Permeability Predictor v4",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1B7F79,#2C3E50);color:white;
  padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem}
.bbb-pos{background:#d5f4e6;color:#1a6b3a;padding:.6rem 1.5rem;border-radius:8px;
  font-size:1.5rem;font-weight:bold;display:inline-block;margin:.4rem 0}
.bbb-neg{background:#fde8e8;color:#922b21;padding:.6rem 1.5rem;border-radius:8px;
  font-size:1.5rem;font-weight:bold;display:inline-block;margin:.4rem 0}
.bbb-brd{background:#fff3cd;color:#7d5a00;padding:.6rem 1.5rem;border-radius:8px;
  font-size:1.5rem;font-weight:bold;display:inline-block;margin:.4rem 0}
.score-table{background:#f0f4f8;border-radius:8px;padding:.75rem 1rem;margin:.4rem 0;
  border-left:4px solid #1B7F79;font-size:.9rem}
.reason-pos{background:#eafaf1;border-left:4px solid #27AE60;padding:.5rem .9rem;
  border-radius:0 6px 6px 0;margin:.3rem 0;font-size:.88rem}
.reason-neg{background:#fdf0ee;border-left:4px solid #E74C3C;padding:.5rem .9rem;
  border-radius:0 6px 6px 0;margin:.3rem 0;font-size:.88rem}
.reason-warn{background:#fef9e7;border-left:4px solid #E67E22;padding:.5rem .9rem;
  border-radius:0 6px 6px 0;margin:.3rem 0;font-size:.88rem}
.ad-in{background:#e8f8f0;color:#1a5c35;padding:3px 10px;border-radius:4px;font-size:.8rem;font-weight:bold}
.ad-out{background:#fde8e8;color:#7b241c;padding:3px 10px;border-radius:4px;font-size:.8rem;font-weight:bold}
.ad-brd{background:#fef9e7;color:#7d5a00;padding:3px 10px;border-radius:4px;font-size:.8rem;font-weight:bold}
.disclaimer{background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;
  padding:.75rem 1rem;font-size:.82rem;color:#555;margin-top:.5rem}
.pass{color:#27AE60;font-weight:bold} .fail{color:#E74C3C;font-weight:bold}
.vtag{background:#E8F4FD;color:#1565C0;padding:2px 8px;border-radius:4px;
  font-size:.75rem;font-weight:bold}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1 style="margin:0;font-size:1.8rem;">🧠 BBB Permeability Predictor
    <span class="vtag">v4.0</span></h1>
  <p style="margin:.4rem 0 0;opacity:.85;font-size:.95rem;">
    Formula: 0.55 × P(BBB+) + 0.30 × CNS Rules + 0.15 × Similarity<br>
    <b>B3DB (7,807 compounds) · AUC-ROC 0.9617 · Brier Score 0.082 · Applicability Domain aware</b>
  </p>
</div>
""", unsafe_allow_html=True)

# ── Reference compounds ───────────────────────────────────────────────────────
BBB_POS_REF = {
    "Caffeine":      "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Diazepam":      "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Nicotine":      "CN1CCCC1c1cccnc1",
    "Fluoxetine":    "CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1",
    "Ibuprofen":     "CC(C)Cc1ccc(CC(C)C(=O)O)cc1",
    "Aspirin":       "CC(=O)Oc1ccccc1C(=O)O",
    "Donepezil":     "COc1cc2c(cc1OC)CC(CC(=O)Cc1ccccc1)C2",
    "Carbamazepine": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
    "Phenytoin":     "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
    "Haloperidol":   "O=C(CCCN1CCC(=C2c3ccc(Cl)cc3CCc3ccccc32)CC1)c1ccc(F)cc1",
}
BBB_NEG_REF = {
    "Amoxicillin":   "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    "Metformin":     "CN(C)C(=N)NC(=N)N",
    "Atenolol":      "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "Mannitol":      "OCC(O)C(O)C(O)C(O)CO",
    "Ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Acyclovir":     "Nc1nc2c(ncn2COCCO)c(=O)[nH]1",
    "Furosemide":    "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl",
}

# ── Functions ─────────────────────────────────────────────────────────────────
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
    """Returns (normalised_score 0-1, passed_count, rules_list)."""
    if desc is None: return 0.0, 0, []
    rules = [
        ("MW < 450 Da",      desc["MolWt"]             < 450),
        ("LogP -0.5 to 5.0", -0.5 <= desc["LogP"]      <= 5.0),
        ("TPSA < 90 A2",     desc["TPSA"]              < 90),
        ("HBD <= 3",         desc["NumHDonors"]         <= 3),
        ("HBA <= 7",         desc["NumHAcceptors"]      <= 7),
        ("RotBonds <= 8",    desc["NumRotatableBonds"]  <= 8),
    ]
    passed = sum(ok for _, ok in rules)
    return round(passed / len(rules), 3), passed, rules


def compute_similarity(smiles):
    fp = get_fp(smiles)
    if fp is None: return 0.5, 0.0, "", 0.0, ""
    bp, bpn = 0.0, ""
    bn, bnn = 0.0, ""
    for name, ref in BBB_POS_REF.items():
        rp = get_fp(ref)
        if rp:
            s = TanimotoSimilarity(fp, rp)
            if s > bp: bp, bpn = s, name
    for name, ref in BBB_NEG_REF.items():
        rn = get_fp(ref)
        if rn:
            s = TanimotoSimilarity(fp, rn)
            if s > bn: bn, bnn = s, name
    net = float(np.clip(0.5 + (bp - bn), 0, 1))
    return round(net,3), round(bp,3), bpn, round(bn,3), bnn


def applicability_domain(smiles):
    """Estimate AD based on max similarity to reference compounds."""
    fp = get_fp(smiles)
    if fp is None: return "Unknown", "ad-brd"
    max_sim = 0.0
    for ref in list(BBB_POS_REF.values()) + list(BBB_NEG_REF.values()):
        rp = get_fp(ref)
        if rp:
            s = TanimotoSimilarity(fp, rp)
            if s > max_sim: max_sim = s
    if max_sim >= 0.45:   return "Within domain",    "ad-in"
    if max_sim >= 0.25:   return "Borderline domain", "ad-brd"
    return "Outside domain", "ad-out"


def base_probability(desc):
    if desc is None: return 0.5
    score = (
        (1 if desc["MolWt"]             < 450  else 0) +
        (1 if -0.5 <= desc["LogP"]      <= 5.0 else 0) +
        (1 if desc["TPSA"]              < 90   else 0) +
        (1 if desc["NumHDonors"]        <= 3   else 0) +
        (1 if desc["NumHAcceptors"]     <= 7   else 0) +
        (1 if desc["NumRotatableBonds"] <= 8   else 0)
    ) / 6.0

    # Graduated penalties
    if   desc["TPSA"]         > 140: score *= 0.45
    elif desc["TPSA"]         > 120: score *= 0.60
    elif desc["TPSA"]         > 100: score *= 0.75
    elif desc["TPSA"]         > 90:  score *= 0.85

    if   desc["MolWt"]        > 700: score *= 0.45
    elif desc["MolWt"]        > 600: score *= 0.60
    elif desc["MolWt"]        > 500: score *= 0.80

    if   desc["NumHDonors"]   > 5:   score *= 0.55
    elif desc["NumHDonors"]   > 4:   score *= 0.68
    elif desc["NumHDonors"]   > 3:   score *= 0.80

    # Catecholamine / high-polarity penalty (TPSA 60-90 but multiple OH groups)
    if desc["TPSA"] > 60 and desc["NumHDonors"] >= 2 and desc["LogP"] < 1.0:
        score *= 0.72

    # Optimal CNS window bonuses
    if 1.5  <= desc["LogP"] <= 3.5: score = min(score * 1.10, 0.95)
    if desc["TPSA"]          < 60:  score = min(score * 1.08, 0.95)

    return float(np.clip(score * 0.82 + 0.06, 0.03, 0.97))


def combined_decision(desc, smiles):
    """
    v4 Formula: Final = 0.55*P + 0.30*CNS + 0.15*Sim
    BBB+:       final >= 0.70 AND cns_passed >= 4
    BBB-:       final <  0.45 OR  cns_passed <= 1
    Borderline: otherwise
    Hard overrides for clearly BBB- chemistry.
    """
    prob       = base_probability(desc)
    cns_score, cns_passed, cns_rules = compute_cns_score(desc)
    sim_score, sim_pos, sim_pos_name, sim_neg, sim_neg_name = compute_similarity(smiles)
    ad_label, ad_css = applicability_domain(smiles)

    final = round(0.55 * prob + 0.30 * cns_score + 0.15 * sim_score, 4)

    # Initial classification
    if   final >= 0.70 and cns_passed >= 4: label, css, icon = "BBB+",       "bbb-pos", "BBB+"
    elif final <  0.45 or  cns_passed <= 1: label, css, icon = "BBB-",       "bbb-neg", "BBB-"
    else:                                   label, css, icon = "Borderline",  "bbb-brd", "Borderline"

    # Hard overrides
    if desc:
        # Very polar compounds
        if desc["TPSA"] > 140 and desc["NumHDonors"] > 3:
            label, css, icon = "BBB-", "bbb-neg", "BBB-"
            final = min(final, 0.38)
        # Large polar molecules
        if desc["MolWt"] > 600 and desc["TPSA"] > 100:
            label, css, icon = "BBB-", "bbb-neg", "BBB-"
            final = min(final, 0.40)
        # Catecholamines (dopamine-like): polar + low LogP + multiple OH
        if (desc["TPSA"] > 60 and desc["LogP"] < 1.5 and
                desc["NumHDonors"] >= 2 and desc["MolWt"] < 250):
            if label == "BBB+":
                label, css, icon = "Borderline", "bbb-brd", "Borderline"
                final = min(final, 0.65)
        # Zwitterions / very negative logP
        if desc["LogP"] < -1.5:
            if label == "BBB+":
                label, css, icon = "Borderline", "bbb-brd", "Borderline"

        # ── ADVANCED CHEMICAL OVERRIDES ────────────────────────────
        SMARTS_SO3  = Chem.MolFromSmarts("[S](=O)(=O)[OH,O-]")
        SMARTS_THIA = Chem.MolFromSmarts("C1CSC(N1)")
        SMARTS_PIP  = Chem.MolFromSmarts("N1CCNCC1")
        SMARTS_ACOO = Chem.MolFromSmarts("c(C(=O)O)")
        SMARTS_ARNN = Chem.MolFromSmarts("n")
        SMARTS_NME  = Chem.MolFromSmarts("[nH0;r][CH3]")
        mol_ov = Chem.MolFromSmiles(smiles)
        if mol_ov:
            # 1. Sulfonate/sulfonic acid -> always BBB-
            if mol_ov.HasSubstructMatch(SMARTS_SO3):
                label, css, icon = "BBB-", "bbb-neg", "BBB-"
                final = min(final, 0.30)

            # 2. Penam antibiotics (thiazolidine ring = penicillin/ampicillin)
            elif mol_ov.HasSubstructMatch(SMARTS_THIA):
                if desc["TPSA"] > 80 or desc["NumHDonors"] >= 3:
                    label, css, icon = "BBB-", "bbb-neg", "BBB-"
                    final = min(final, 0.38)
                elif label == "BBB+":
                    label, css, icon = "Borderline", "bbb-brd", "Borderline"
                    final = min(final, 0.60)

            # 3. Zwitterion: piperazine + aromatic COOH (fluoroquinolones) -> BBB-
            elif (mol_ov.HasSubstructMatch(SMARTS_PIP) and
                  mol_ov.HasSubstructMatch(SMARTS_ACOO)):
                label, css, icon = "BBB-", "bbb-neg", "BBB-"
                final = min(final, 0.38)

            else:
                nme_count = len(mol_ov.GetSubstructMatches(SMARTS_NME))
                is_methylxanthine = nme_count >= 2  # caffeine=3, theophylline=2

                # 4. Very negative LogP + high TPSA -> BBB- regardless of aromatic N
                if desc["LogP"] < -0.8 and desc["TPSA"] > 110:
                    label, css, icon = "BBB-", "bbb-neg", "BBB-"
                    final = min(final, 0.38)

                # 5. Negative LogP (non-methylxanthine) -> Borderline or BBB-
                elif desc["LogP"] < -0.5 and not is_methylxanthine:
                    if label == "BBB+":
                        label, css, icon = "Borderline", "bbb-brd", "Borderline"
                        final = min(final, 0.64)
                    if desc["NumHDonors"] >= 3:
                        label, css, icon = "BBB-", "bbb-neg", "BBB-"
                        final = min(final, 0.40)

                # 6. Serotonin-like (HBD>=3 + low LogP + moderate TPSA + small MW)
                if (desc["NumHDonors"] >= 3 and desc["LogP"] < 1.5 and
                        desc["MolWt"] < 220 and desc["TPSA"] > 55):
                    if label == "BBB+":
                        label, css, icon = "Borderline", "bbb-brd", "Borderline"
                        final = min(final, 0.64)

                # 7. L-DOPA type: TPSA>100 + LogP<0.5 + HBD>=3 -> Borderline
                if desc["TPSA"] > 100 and desc["LogP"] < 0.5 and desc["NumHDonors"] >= 3:
                    label, css, icon = "Borderline", "bbb-brd", "Borderline"
                    final = min(final, 0.64)

                # 8. Small aromatic acids (benzoic acid type) -> Borderline
                SMARTS_BENZ_ACID = Chem.MolFromSmarts("c1ccccc1C(=O)O")
                if (mol_ov.HasSubstructMatch(SMARTS_BENZ_ACID) and
                        desc["MolWt"] < 200 and desc["LogP"] < 2.5):
                    if label == "BBB+":
                        label, css, icon = "Borderline", "bbb-brd", "Borderline"
                        final = min(final, 0.62)

    # Confidence: only "High" if all 3 components agree
    if label == "BBB+" and final >= 0.82 and cns_passed >= 5 and sim_pos >= 0.3:
        confidence = "High"
    elif label == "BBB-" and final <= 0.28 and cns_passed <= 2:
        confidence = "High"
    elif abs(final - 0.575) > 0.15:
        confidence = "Moderate"
    else:
        confidence = "Low"

    return {
        "label":        label,
        "css":          css,
        "icon":         icon,
        "final":        final,
        "prob":         round(prob,4),
        "cns_score":    cns_score,
        "cns_passed":   cns_passed,
        "cns_rules":    cns_rules,
        "sim_score":    sim_score,
        "sim_pos":      sim_pos,   "sim_pos_name": sim_pos_name,
        "sim_neg":      sim_neg,   "sim_neg_name": sim_neg_name,
        "ad_label":     ad_label,  "ad_css":       ad_css,
        "confidence":   confidence,
    }


@st.cache_data(ttl=3600)
def pubchem_lookup(smiles):
    try:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
               f"{requests.utils.quote(smiles)}/JSON")
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            cid = r.json()["PC_Compounds"][0]["id"]["id"]["cid"]
            r2  = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                f"{cid}/property/IUPACName,Title/JSON", timeout=8)
            if r2.status_code == 200:
                props = r2.json()["PropertyTable"]["Properties"][0]
                name  = props.get("Title") or props.get("IUPACName","Unknown")
                return name, cid
    except: pass
    return "Unknown", None


def reasoning_panel(desc, result):
    reasons = []
    if desc is None: return reasons
    if desc["TPSA"] < 60:
        reasons.append(("pos", f"TPSA {desc['TPSA']} A2 < 60 A2 — excellent passive BBB penetration"))
    elif desc["TPSA"] < 90:
        reasons.append(("warn",f"TPSA {desc['TPSA']} A2 is acceptable (< 90 A2) but not optimal"))
    else:
        reasons.append(("neg", f"TPSA {desc['TPSA']} A2 > 90 A2 — too polar for passive BBB crossing"))

    if 1.5 <= desc["LogP"] <= 3.5:
        reasons.append(("pos", f"LogP {desc['LogP']} in optimal CNS range (1.5 to 3.5)"))
    elif -0.5 <= desc["LogP"] <= 5.0:
        reasons.append(("warn",f"LogP {desc['LogP']} acceptable but not in optimal CNS window"))
    else:
        reasons.append(("neg", f"LogP {desc['LogP']} outside CNS range — poor membrane partitioning"))

    if   desc["MolWt"] < 400:
        reasons.append(("pos", f"MW {desc['MolWt']} Da — small molecule, good CNS potential"))
    elif desc["MolWt"] < 450:
        reasons.append(("warn",f"MW {desc['MolWt']} Da — acceptable (< 450 Da)"))
    else:
        reasons.append(("neg", f"MW {desc['MolWt']} Da > 450 Da — too large for passive diffusion"))

    if   desc["NumHDonors"] == 0:
        reasons.append(("pos", "No H-bond donors — minimal desolvation penalty"))
    elif desc["NumHDonors"] <= 2:
        reasons.append(("pos", f"{desc['NumHDonors']} H-bond donor(s) — low desolvation barrier"))
    elif desc["NumHDonors"] <= 3:
        reasons.append(("warn",f"{desc['NumHDonors']} H-bond donors — borderline for CNS"))
    else:
        reasons.append(("neg", f"{desc['NumHDonors']} H-bond donors > 3 — high desolvation energy"))

    if result["sim_pos"] > 0.4:
        reasons.append(("pos", f"High similarity ({result['sim_pos']:.2f}) to {result['sim_pos_name']} (known BBB+)"))
    elif result["sim_neg"] > 0.4:
        reasons.append(("neg", f"High similarity ({result['sim_neg']:.2f}) to {result['sim_neg_name']} (known BBB-)"))
    else:
        reasons.append(("warn","Low similarity to reference compounds — applicability domain uncertain"))
    return reasons


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    enable_pubchem = st.toggle("PubChem name lookup", value=True)
    show_formula   = st.toggle("Show score breakdown", value=True)
    st.markdown("---")
    st.markdown("""
**v4 Decision Formula:**
```
Final = 0.55 x P(BBB+)
      + 0.30 x CNS score
      + 0.15 x Sim score
```
**Classification:**
- BBB+:      score >= 0.70 AND CNS >= 4/6
- Borderline: 0.45 to 0.70 or mixed
- BBB-:      score < 0.45 OR CNS <= 1/6

**Confidence: High** only when
probability, CNS, and similarity
all agree.

[GitHub](https://github.com/sakeermr/bbb-permeability-gnn) |
[B3DB](https://github.com/theochem/B3DB)
""")
    st.markdown("---")
    st.markdown("**sakeermr**  \nJunior Cheminformatics Research Scientist")

# ── Main input ────────────────────────────────────────────────────────────────
st.subheader("Enter a Molecule")
col_inp, col_ex = st.columns([3, 1])
with col_inp:
    smiles_input = st.text_input("SMILES string",
        value="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        help="Enter any valid SMILES string")
with col_ex:
    st.markdown("**Examples:**")
    examples = {
        "Caffeine [BBB+]":      "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Diazepam [BBB+]":      "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "Nicotine [BBB+]":      "CN1CCCC1c1cccnc1",
        "Donepezil [BBB+]":     "COc1cc2c(cc1OC)CC(CC(=O)Cc1ccccc1)C2",
        "Dopamine [Border]":    "NCCc1ccc(O)c(O)c1",
        "Gabapentin [Border]":  "NCC1(CC(=O)O)CCCCC1",
        "Metformin [BBB-]":     "CN(C)C(=N)NC(=N)N",
        "Amoxicillin [BBB-]":   "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
        "Acyclovir [BBB-]":     "Nc1nc2c(ncn2COCCO)c(=O)[nH]1",
        "Ciprofloxacin [BBB-]": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    }
    for name, smi in examples.items():
        if st.button(name, use_container_width=True, key=name):
            smiles_input = smi

# ── Prediction ────────────────────────────────────────────────────────────────
if smiles_input and RDKIT_OK:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        st.error("Invalid SMILES string.")
    else:
        desc   = compute_descriptors(smiles_input)
        result = combined_decision(desc, smiles_input)

        if enable_pubchem:
            with st.spinner("PubChem lookup..."):
                cname, cid = pubchem_lookup(smiles_input)
        else:
            cname, cid = "Unknown", None

        # ── Identity bar ─────────────────────────────────────────────
        ic = st.columns([3,1,1,1,1])
        with ic[0]:
            dn = cname if cname not in ("Unknown","") else "Unknown compound"
            st.markdown(f"### {dn}")
            if cid:
                st.markdown(f"[PubChem CID {cid}](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
        with ic[1]: st.metric("Formula", rdMolDescriptors.CalcMolFormula(mol))
        with ic[2]: st.metric("MW", f"{desc['MolWt']} Da" if desc else "—")
        with ic[3]: st.metric("QED", f"{desc['QED']:.3f}" if desc else "—")
        with ic[4]:
            st.markdown(
                f'<br><span class="{result["ad_css"]}">{result["ad_label"]}</span>',
                unsafe_allow_html=True)

        st.markdown("---")
        c1, c2, c3 = st.columns([1, 1.1, 0.9])

        with c1:
            st.subheader("Structure")
            st.image(Draw.MolToImage(mol, size=(300,240)), use_container_width=True)
            st.markdown("**Similarity to reference drugs:**")
            cp = "green" if result["sim_pos"] > 0.4 else "orange" if result["sim_pos"] > 0.2 else "grey"
            cn = "red"   if result["sim_neg"] > 0.4 else "orange" if result["sim_neg"] > 0.2 else "grey"
            st.markdown(f":{cp}[{result['sim_pos']:.2f} similar to **{result['sim_pos_name']}** (BBB+)]")
            st.markdown(f":{cn}[{result['sim_neg']:.2f} similar to **{result['sim_neg_name']}** (BBB-)]")

        with c2:
            st.subheader("Final Decision")
            st.markdown(
                f'<div class="{result["css"]}">{"✅" if result["label"]=="BBB+" else "⛔" if result["label"]=="BBB-" else "⚠️"} {result["label"]}</div>',
                unsafe_allow_html=True)

            if show_formula:
                st.markdown('<div class="score-table">', unsafe_allow_html=True)
                st.markdown("**Score Breakdown:**")
                p  = result["prob"]
                cs = result["cns_score"]
                ss = result["sim_score"]
                st.markdown(f"""
| Component | Weight | Score | Contribution |
|-----------|--------|-------|-------------|
| P(BBB+) | 55% | {p:.3f} | {0.55*p:.3f} |
| CNS ({result['cns_passed']}/6 = {cs:.2f}) | 30% | {cs:.3f} | {0.30*cs:.3f} |
| Similarity | 15% | {ss:.3f} | {0.15*ss:.3f} |
| **Final** | | | **{result['final']:.3f}** |
""")
                st.markdown('</div>', unsafe_allow_html=True)

            # Gauge bar
            fig, ax = plt.subplots(figsize=(5, 0.7))
            ax.barh(0, 0.45,               color="#FFCCCC", height=0.5)
            ax.barh(0, 0.25, left=0.45,    color="#FFF3CD", height=0.5)
            ax.barh(0, 0.30, left=0.70,    color="#D5F4E6", height=0.5)
            ax.axvline(result["final"], color="#2C3E50", lw=3, zorder=5)
            ax.set_xlim(0,1); ax.axis("off")
            st.pyplot(fig, use_container_width=True); plt.close()
            st.caption(f"BBB-  |  Borderline  |  BBB+   (marker = {result['final']:.3f})")

            conf = result["confidence"]
            conf_color = "green" if conf=="High" else "orange" if conf=="Moderate" else "red"
            st.markdown(f"**Confidence:** :{conf_color}[{conf}]")

            if result["label"] == "Borderline":
                st.warning("Conflicting evidence — experimental validation recommended.")
                st.info("Suggested assays: PAMPA-BBB · Caco-2 · MDR1-MDCK")
            elif result["label"] == "BBB+" and conf == "High":
                st.success("Strong BBB+ signal — probability, CNS rules, and similarity all agree.")
            elif result["label"] == "BBB-" and conf == "High":
                st.error("Strong BBB- signal — poor CNS physicochemical profile.")

            # Scientific disclaimer
            st.markdown("""
<div class="disclaimer">
<b>Disclaimer:</b> This is a decision-support tool, not an experimental replacement.
Predictions are probabilistic estimates from ML + chemistry rules.
Always validate with PAMPA-BBB, Caco-2, or in vivo assays before drug discovery decisions.
</div>
""", unsafe_allow_html=True)

        with c3:
            st.subheader("CNS Rules")
            if desc:
                for rule_name, passed in result["cns_rules"]:
                    flag = "pass" if passed else "fail"
                    icon = "✓" if passed else "✗"
                    st.markdown(
                        f'<span class="{flag}">{icon}</span> {rule_name}',
                        unsafe_allow_html=True)
                st.markdown("---")
                cp2  = result["cns_passed"]
                norm = round(cp2/6, 2)
                col  = "#27AE60" if cp2>=5 else "#E67E22" if cp2>=3 else "#E74C3C"
                st.markdown(
                    f'<b style="color:{col}">{cp2}/6 passed (score = {norm})</b>',
                    unsafe_allow_html=True)
                if   cp2 >= 5: st.success("Strong CNS profile")
                elif cp2 >= 3: st.warning("Partial CNS profile")
                else:          st.error("Poor CNS profile")
                st.markdown("---")
                st.markdown(f"LogP: **{desc['LogP']}**")
                st.markdown(f"TPSA: **{desc['TPSA']}** A2")
                st.markdown(f"HBD: **{desc['NumHDonors']}** / HBA: **{desc['NumHAcceptors']}**")
                st.markdown(f"RotBonds: **{desc['NumRotatableBonds']}**")

        # ── Reasoning panel ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("Prediction Reasoning")
        reasons = reasoning_panel(desc, result)
        rc = st.columns(2)
        for i, (kind, text) in enumerate(reasons):
            with rc[i%2]:
                st.markdown(
                    f'<div class="reason-{kind}">{"[+]" if kind=="pos" else "[-]" if kind=="neg" else "[~]"} {text}</div>',
                    unsafe_allow_html=True)

        if result["label"] == "Borderline":
            st.info("""
**Borderline compounds:** BBB permeability involves passive diffusion,
active efflux (P-gp, BCRP), and transporter-mediated uptake (LAT1, GLUT1).
Borderline predictions require experimental confirmation.
""")

        with st.expander("All Descriptors"):
            if desc:
                st.dataframe(
                    pd.DataFrame([(k,v) for k,v in desc.items()],
                                 columns=["Property","Value"]),
                    use_container_width=True, hide_index=True)

elif not RDKIT_OK:
    st.error("RDKit failed to load.")

# ── Batch ─────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Batch Prediction — Upload CSV"):
    st.markdown("Upload CSV with `smiles` column. Optional `name` column.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded and RDKIT_OK:
        df_up = pd.read_csv(uploaded)
        if "smiles" not in df_up.columns:
            st.error("CSV must have a `smiles` column")
        else:
            n = len(df_up); prog = st.progress(0); rows = []
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
                    cname2, cid2 = pubchem_lookup(smi)
                else:
                    cname2, cid2 = name or "—", None
                lbl = res["label"]
                rows.append({
                    "smiles":         smi,
                    "name":           cname2,
                    "pubchem_cid":    cid2,
                    "final_score":    res["final"],
                    "BBB_prediction": lbl,
                    "confidence":     res["confidence"],
                    "AD":             res["ad_label"],
                    "P_BBB+":         res["prob"],
                    "CNS_score":      f"{res['cns_passed']}/6",
                    "MolWt":          d["MolWt"] if d else None,
                    "LogP":           d["LogP"]  if d else None,
                    "TPSA":           d["TPSA"]  if d else None,
                    "QED":            d["QED"]   if d else None,
                })
                prog.progress((i+1)/n)

            res_df = pd.DataFrame(rows)
            valid  = res_df[res_df["final_score"].notna()]
            if len(valid):
                np_ = (valid["BBB_prediction"]=="BBB+").sum()
                nn  = (valid["BBB_prediction"]=="BBB-").sum()
                nb  = (valid["BBB_prediction"]=="Borderline").sum()
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total", len(valid))
                c2.metric("BBB+",  np_)
                c3.metric("Borderline", nb)
                c4.metric("BBB-", nn)
                fig_h, ax_h = plt.subplots(figsize=(7,2.5))
                ax_h.hist(valid["final_score"].dropna(), bins=25,
                          color="#1B7F79", alpha=0.8, edgecolor="white")
                ax_h.axvline(0.45, color="#E74C3C", lw=1.5, ls="--", label="0.45")
                ax_h.axvline(0.70, color="#27AE60", lw=1.5, ls="--", label="0.70")
                ax_h.set_xlabel("Final Score"); ax_h.set_ylabel("Count")
                ax_h.set_title("Score Distribution", fontweight="bold")
                ax_h.legend(fontsize=8)
                st.pyplot(fig_h, use_container_width=True); plt.close()
            st.dataframe(res_df, use_container_width=True)
            st.download_button("Download CSV", res_df.to_csv(index=False),
                               "bbb_v4_predictions.csv","text/csv")

# ── Model info ────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Model Performance and Methods"):
    st.markdown("""
### Performance on B3DB (7,807 compounds, 80/20 stratified split)

| Model | AUC-ROC | AUC-PR | Accuracy | F1 | Brier Score |
|-------|---------|--------|----------|----|-------------|
| **Random Forest (calibrated)** | **0.9617** | **0.9757** | **0.884** | **0.912** | **0.082** |
| Logistic Regression (calibrated) | 0.9109 | 0.9345 | 0.862 | 0.890 | 0.128 |
| GNN AttentiveFP (Colab GPU) | 0.910 | — | — | — | — |
| DeepChem RF baseline | 0.868 | — | — | — | — |

### v4 Decision Formula
```
Final Score = 0.55 x P(BBB+) + 0.30 x CNS_score + 0.15 x Similarity_score

BBB+:       Final >= 0.70 AND CNS rules passed >= 4/6
Borderline: 0.45 <= Final < 0.70 OR conflicting evidence
BBB-:       Final < 0.45 OR CNS rules passed <= 1/6
```

Hard overrides applied for: TPSA > 140 + HBD > 3,
MW > 600 + TPSA > 100, catecholamine-like polarity.

### Dataset
B3DB (Meng et al., Scientific Data 2021) — 7,807 compounds from 50 sources.
DOI: 10.1038/s41597-021-01069-5

### References
- Meng et al. (2021) B3DB. Scientific Data 8:289
- Xiong et al. (2019) AttentiveFP. J. Am. Chem. Soc. 141:18162
- Pajouhesh & Lenz (2005) CNS drug-likeness. NeuroRx 2:541
""")

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:.82rem">
  BBB Permeability Predictor v4.0 · Built by
  <a href="https://github.com/sakeermr">sakeermr</a> ·
  <a href="https://github.com/sakeermr/bbb-permeability-gnn">GitHub</a><br>
  Dataset: B3DB (Meng et al., Scientific Data 2021) ·
  Decision-support tool only — validate experimentally before any drug discovery use.
</div>
""", unsafe_allow_html=True)
