# Dataset — Blood-Brain Barrier Permeability (BBBP)

## Overview
87 curated compounds with experimentally validated BBB permeability labels.

| Class | Count | Percentage | Description |
|-------|-------|-----------|-------------|
| BBB+ (p_np = 1) | 47 | 54.0% | CNS-permeable drugs |
| BBB- (p_np = 0) | 40 | 46.0% | CNS-impermeable drugs |
| **Total** | **87** | — | — |

## File descriptions

| File | Description |
|------|-------------|
| `raw/bbbp.csv` | Original SMILES + labels. Columns: `name`, `smiles`, `p_np` |
| `processed/bbbp_clean.csv` | After RDKit validation + descriptor computation + scaffold split labels |

## Columns (processed file)

| Column | Type | Description |
|--------|------|-------------|
| `name` | str | Drug/compound name |
| `smiles` | str | Canonical SMILES (RDKit-validated) |
| `p_np` | int | Label: 1=BBB+ (permeable), 0=BBB- (impermeable) |
| `scaffold` | str | Bemis-Murcko scaffold SMILES |
| `split` | str | `train` or `test` (scaffold-based 80/20) |
| `MolWt` | float | Molecular weight (Da) |
| `LogP` | float | Calculated octanol-water partition coefficient |
| `NumHDonors` | int | Number of H-bond donors |
| `NumHAcceptors` | int | Number of H-bond acceptors |
| `TPSA` | float | Topological polar surface area (Å²) |
| `NumRotatableBonds` | int | Number of rotatable bonds |
| `NumAromaticRings` | int | Number of aromatic rings |
| `QED` | float | Quantitative estimate of drug-likeness (0–1) |
| `FractionCSP3` | float | Fraction of sp3 carbons |
| ... | ... | See `preprocessing.py` for full descriptor list |

## BBB+ Compounds (selected)
CNS-active drugs confirmed to cross the blood-brain barrier:
- Benzodiazepines: Diazepam, Lorazepam, Alprazolam, Temazepam, Midazolam
- Antidepressants: Fluoxetine, Sertraline, Paroxetine, Venlafaxine, Bupropion
- Antipsychotics: Haloperidol, Clozapine, Olanzapine, Quetiapine, Risperidone
- Analgesics: Morphine, Fentanyl, Tramadol
- Stimulants: Caffeine, Nicotine, Amphetamine, Methylphenidate
- Anticonvulsants: Carbamazepine, Phenobarbital, Pregabalin, Levetiracetam
- Other CNS: Donepezil (Alzheimer's), Memantine, Propranolol, Melatonin

## BBB- Compounds (selected)
Drugs confirmed NOT to cross the blood-brain barrier:
- Antibiotics: Amoxicillin, Ampicillin, Ciprofloxacin, Doxycycline, Erythromycin
- Antihypertensives: Atenolol, Metoprolol, Lisinopril, Losartan, Amlodipine
- Statins: Atorvastatin, Pravastatin
- Antidiabetics: Metformin
- Antivirals: Acyclovir, Oseltamivir, Remdesivir
- Sugars: Glucose, Mannitol, Sucrose
- Anticancer: Imatinib, Methotrexate, Doxorubicin

## Data Quality Notes
- All SMILES validated with RDKit — invalid structures excluded
- Stereochemistry preserved where present
- Labels based on published experimental data and clinical knowledge
- No duplicates (verified by canonical SMILES comparison)

## Extending the Dataset
For a larger, publication-quality dataset, use:
```python
# MoleculeNet BBBP (2,050 compounds)
import deepchem as dc
tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer='GraphConv', splitter='scaffold')

# TDC BBBP (2,030 compounds)
from tdc.single_pred import ADME
data = ADME(name='BBB_Martins')
```

## Citation
If you use this dataset, please also cite the original BBBP benchmark:
```
Martins IF et al. (2012) A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling.
J. Chem. Inf. Model. 52(6), 1686–1697. DOI: 10.1021/ci300124c
```
