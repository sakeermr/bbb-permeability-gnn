<div align="center">

# 🧠 BBB Permeability Predictor
### Calibrated multi-model prediction of Blood-Brain Barrier permeability for CNS drug discovery

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![RDKit](https://img.shields.io/badge/RDKit-2023-green?style=flat-square)](https://rdkit.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/Tests-23%20passed-brightgreen?style=flat-square)](tests/)
[![CI](https://github.com/sakeermr/bbb-permeability-gnn/actions/workflows/ci.yml/badge.svg)](https://github.com/sakeermr/bbb-permeability-gnn/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bbb-permeability-gnn-smz6lvxqzkw9qtd2mjyfff.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**sakeermr** · Junior Cheminformatics Research Scientist  
[LinkedIn](https://linkedin.com/in/sakeermr) · [GitHub](https://github.com/sakeermr) · [Live App](https://bbb-permeability-gnn-smz6lvxqzkw9qtd2mjyfff.streamlit.app/) · [Technical Report](docs/BBB_Technical_Report.md)

</div>

---

## Overview

Blood-brain barrier (BBB) permeability is one of the most critical ADMET properties in CNS drug discovery. This project builds a **calibrated, interpretable prediction system** that combines machine learning with established medicinal chemistry knowledge to estimate BBB permeability from molecular SMILES strings.

The system uses a three-component decision formula:

```
Final Score = 0.50 × P(BBB+) + 0.35 × CNS Rules + 0.15 × Similarity
```

with ten chemistry-based override rules for structural classes where ML probability alone is insufficient (sulfonates, penam antibiotics, fluoroquinolones, catecholamines, etc.).

**Key result:** AUC-ROC **0.9617** on B3DB test set (+9.2% vs DeepChem baseline) · **15/15** benchmark molecules correctly classified (100%)

---

## Results

### Model Performance on B3DB (7,807 compounds · stratified 80/20 split)

| Model | AUC-ROC | AUC-PR | Accuracy | F1 | Brier Score |
|-------|---------|--------|----------|----|-------------|
| **Random Forest (calibrated)** | **0.9617** | **0.9757** | **0.884** | **0.912** | **0.082** |
| Logistic Regression (calibrated) | 0.9109 | 0.9345 | 0.862 | 0.890 | 0.128 |
| GNN AttentiveFP (Colab GPU) | 0.910 | — | — | — | — |
| DeepChem RF baseline | 0.868 | — | — | — | — |

> All results use **stratified 80/20 split**. Brier Score of 0.082 indicates well-calibrated probabilities.

### Benchmark Validation — 15 Curated Molecules (100% accuracy)

| Compound | Expected | Predicted | Score | CNS |
|----------|----------|-----------|-------|-----|
| Caffeine | BBB+ | ✅ BBB+ | 0.837 | 5/6 |
| Diazepam | BBB+ | ✅ BBB+ | 0.928 | 6/6 |
| Nicotine | BBB+ | ✅ BBB+ | 0.928 | 6/6 |
| Fluoxetine | BBB+ | ✅ BBB+ | 0.928 | 6/6 |
| Morphine | BBB+ | ✅ BBB+ | 0.928 | 6/6 |
| Atropine | BBB+ | ✅ BBB+ | 0.846 | 6/6 |
| Dopamine | Borderline | ✅ Borderline | 0.697 | 6/6 |
| Serotonin | Borderline | ✅ Borderline | 0.691 | 6/6 |
| L-DOPA | Borderline | ✅ Borderline | 0.466 | 4/6 |
| Gabapentin | Borderline | ✅ Borderline | 0.727 | 6/6 |
| Ampicillin | BBB- | ✅ BBB- | 0.343 | 4/6 |
| Acyclovir | BBB- | ✅ BBB- | 0.335 | 4/6 |
| Metformin | BBB- | ✅ BBB- | 0.349 | 4/6 |
| Benzenesulfonic acid | BBB- | ✅ BBB- | 0.845* | 6/6 |
| Ciprofloxacin | BBB- | ✅ BBB- | 0.778* | 6/6 |

*Score before chemical override applied.

---

## Features

- **3-tier classification** — BBB+ / Borderline / BBB- (reduces false certainty)
- **Calibrated ensemble** — Random Forest + Logistic Regression, Brier Score 0.082
- **10 chemical override rules** — sulfonates, penams, fluoroquinolones, catecholamines
- **Methylxanthine-aware** — caffeine/theophylline LogP penalty excluded correctly
- **Applicability domain** — Within / Borderline / Outside domain label per prediction
- **PubChem name lookup** — auto-identifies compound name and CID from SMILES
- **Prediction reasoning panel** — explains each classification in chemical terms
- **Structural similarity** — Tanimoto comparison to 17 known BBB+/BBB- reference drugs
- **Batch prediction** — upload CSV, get predictions + probability distribution plot
- **23 unit tests** — GitHub Actions CI passing

---

## Project Structure

```
bbb-permeability-gnn/
├── app/
│   └── streamlit_app.py          # v6.1 interactive web app (live on Streamlit Cloud)
├── data/
│   ├── raw/
│   │   └── B3DB_classification.tsv   # 7,807 compounds (Meng et al. 2021)
│   └── processed/
│       └── b3db_clean.csv            # Validated + descriptors computed
├── src/
│   ├── data/preprocessing.py         # RDKit descriptors, fingerprints, scaffold split
│   ├── models/
│   │   ├── models.py                 # RF, GBT, SVM, LR baselines
│   │   └── gnn_attentivefp.py        # AttentiveFP GNN (PyTorch Geometric, run on Colab)
│   └── evaluation/evaluate.py        # Metrics, ROC curves, SHAP, all figures
├── scripts/
│   ├── train_baselines.py            # Train all models + generate figures
│   └── predict.py                    # CLI prediction from SMILES or CSV
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA + descriptor analysis
│   └── 02_gnn_training.ipynb         # AttentiveFP GNN (Google Colab, GPU)
├── docs/
│   └── BBB_Technical_Report.md       # Full technical report + results discussion
├── figures/                          # Publication-quality output figures
├── results/                          # Model metrics + saved models
├── tests/
│   └── test_pipeline.py              # 23 unit tests (100% passing)
├── configs/config.yaml               # Hyperparameters
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option 1 — Live Web App (no install)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bbb-permeability-gnn-smz6lvxqzkw9qtd2mjyfff.streamlit.app/)

### Option 2 — Local Installation

```bash
git clone https://github.com/sakeermr/bbb-permeability-gnn.git
cd bbb-permeability-gnn
pip install -r requirements.txt
python scripts/train_baselines.py
```

### Option 3 — Predict a Single Molecule

```bash
# BBB+ example (Caffeine)
python scripts/predict.py --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C"

# BBB- example (Amoxicillin)
python scripts/predict.py --smiles "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O"

# Batch predict
python scripts/predict.py --csv molecules.csv --output results/predictions.csv
```

### Option 4 — GNN on Google Colab (GPU)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sakeermr/bbb-permeability-gnn/blob/main/notebooks/02_gnn_training.ipynb)

---

## Methods

### Dataset
**B3DB** (Meng et al., *Scientific Data* 2021) — 7,807 compounds with experimentally validated BBB permeability labels compiled from 50 peer-reviewed sources. DOI: [10.1038/s41597-021-01069-5](https://doi.org/10.1038/s41597-021-01069-5)

- 4,956 BBB+ (CNS-permeable)
- 2,851 BBB- (non-permeable)
- Source: [github.com/theochem/B3DB](https://github.com/theochem/B3DB)

### Decision Formula
```
Final Score = 0.50 × P(BBB+) + 0.35 × CNS_score + 0.15 × Similarity_score

BBB+:       Final Score ≥ 0.72  AND  CNS rules ≥ 4/6
Borderline: 0.42 ≤ Final Score < 0.72
BBB-:       Final Score < 0.42  OR   CNS rules ≤ 1/6
```

### CNS Drug-Likeness Rules (Pajouhesh & Lenz, 2005)

| Property | BBB+ Criterion | Lipinski Ro5 |
|----------|---------------|-------------|
| Molecular Weight | < 450 Da | < 500 Da |
| LogP | −0.5 to 5.0 | < 5.0 |
| TPSA | < 90 Å² | < 140 Å² |
| H-Bond Donors | ≤ 3 | ≤ 5 |
| H-Bond Acceptors | ≤ 7 | ≤ 10 |
| Rotatable Bonds | ≤ 8 | ≤ 10 |

---

## Reproducibility

```bash
# Run all unit tests
pytest tests/ -v

# Reproduce all results from scratch
python scripts/train_baselines.py

# All figures saved to figures/
# All metrics saved to results/all_model_metrics.csv
```

Random seeds fixed at 42. Results fully reproducible.

---

## Citation

```bibtex
@misc{sakeermr2026bbb,
  title   = {BBB Permeability Predictor v6.1: Calibrated Multi-Model Prediction
             of Blood-Brain Barrier Permeability for CNS Drug Discovery},
  author  = {sakeermr},
  year    = {2026},
  url     = {https://github.com/sakeermr/bbb-permeability-gnn},
  note    = {Dataset: B3DB (Meng et al., Scientific Data 2021)}
}
```

### Dataset Citation
```bibtex
@article{Meng2021B3DB,
  author  = {Meng, Fanwang and Xi, Yang and Huang, Jinfeng and Ayers, Paul W.},
  title   = {A curated diverse molecular database of blood-brain barrier permeability
             with chemical descriptors},
  journal = {Scientific Data},
  volume  = {8},
  pages   = {289},
  year    = {2021},
  doi     = {10.1038/s41597-021-01069-5}
}
```

---

## Related Work

- Meng et al. (2021) B3DB dataset. *Scientific Data* — gold-standard BBB benchmark
- Xiong et al. (2019) AttentiveFP. *J. Am. Chem. Soc.* — GNN architecture used
- Pajouhesh & Lenz (2005) CNS drug-likeness rules. *NeuroRx* — CNS rules reference
- Wu et al. (2018) MoleculeNet. *Chem. Sci.* — benchmark framework

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Contact

**sakeermr**  
Junior Cheminformatics Research Scientist  
🔗 [linkedin.com/in/sakeermr](https://linkedin.com/in/sakeermr)  
🐙 [github.com/sakeermr](https://github.com/sakeermr)

*Open to research collaborations and consulting in computational drug discovery.*
