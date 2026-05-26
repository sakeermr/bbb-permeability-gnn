"""
BBB Permeability Predictor — Evaluation & Visualisation
Author: sakeermr
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score,
)
import warnings
warnings.filterwarnings('ignore')

# Publication-quality style
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "xtick.major.width":0.8,
    "ytick.major.width":0.8,
})

PALETTE = {
    "Random Forest":        "#1B7F79",
    "Gradient Boosting":    "#E07B39",
    "SVM (RBF)":            "#7C5CBF",
    "Logistic Regression":  "#C0392B",
    "GNN (AttentiveFP)":    "#2C3E50",
}


# ── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute all classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC-ROC":   round(roc_auc_score(y_true, y_prob), 4),
        "AUC-PR":    round(average_precision_score(y_true, y_prob), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "F1":        round(f1_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


# ── Figure 1: ROC curves comparison ─────────────────────────────────────────

def plot_roc_curves(results: dict, save_path: str = "figures/fig1_roc_curves.png"):
    """
    results = {
        "Model Name": {"y_true": array, "y_prob": array},
        ...
    }
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc = roc_auc_score(data["y_true"], data["y_prob"])
        color = PALETTE.get(name, "#333333")
        lw = 2.5 if "GNN" in name else 1.8
        ls = "-"   if "GNN" in name else "--"
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.03, color="gray")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — BBB Permeability Models", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── Figure 2: Confusion matrix ───────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_prob, model_name: str = "Best Model",
                           save_path: str = "figures/fig2_confusion_matrix.png"):
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                linewidths=0.5, ax=ax, cbar=False)

    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm_pct[i,j] > 55 else "black")

    ax.set_xticklabels(["BBB−\nPredicted", "BBB+\nPredicted"], fontsize=10)
    ax.set_yticklabels(["BBB−\nActual", "BBB+\nActual"], fontsize=10, rotation=0)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── Figure 3: Metrics comparison bar chart ──────────────────────────────────

def plot_metrics_comparison(all_metrics: dict,
                             save_path: str = "figures/fig3_metrics_comparison.png"):
    """
    all_metrics = {
        "Model Name": {"AUC-ROC": 0.91, "Accuracy": 0.87, ...},
        ...
    }
    """
    metric_cols = ["AUC-ROC", "Accuracy", "F1", "Precision", "Recall"]
    models = list(all_metrics.keys())
    x = np.arange(len(metric_cols))
    width = 0.75 / len(models)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, model in enumerate(models):
        vals = [all_metrics[model].get(m, 0) for m in metric_cols]
        bars = ax.bar(x + i * width - (len(models)-1)*width/2, vals,
                      width=width*0.9,
                      color=PALETTE.get(model, f"C{i}"),
                      label=model, alpha=0.88, edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="500")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("Performance Comparison — All Models", fontsize=13, fontweight="bold", pad=12)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── Figure 4: Chemical space PCA ────────────────────────────────────────────

def plot_chemical_space(df: pd.DataFrame,
                         save_path: str = "figures/fig4_chemical_space.png"):
    """PCA of Morgan fingerprints coloured by BBB label + split."""
    from rdkit.Chem import AllChem
    from rdkit import Chem
    from sklearn.decomposition import PCA

    fps, valid_idx = [], []
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fps.append(list(fp))
            valid_idx.append(i)

    X   = np.array(fps)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    labels = df.loc[valid_idx, "p_np"].values
    split  = df.loc[valid_idx, "split"].values if "split" in df.columns else ["train"]*len(valid_idx)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: colour by BBB label
    colors = ["#E74C3C" if l == 0 else "#2ECC71" for l in labels]
    for ax, (col_arr, title, legend_items) in zip(axes, [
        (colors, "Chemical Space — BBB Class",
         [mpatches.Patch(color="#2ECC71", label="BBB+ (permeable)"),
          mpatches.Patch(color="#E74C3C", label="BBB− (impermeable)")]),
        (["#3498DB" if s == "train" else "#E67E22" for s in split],
         "Chemical Space — Train / Test Split",
         [mpatches.Patch(color="#3498DB", label="Train"),
          mpatches.Patch(color="#E67E22", label="Test")]),
    ]):
        ax.scatter(coords[:, 0], coords[:, 1], c=col_arr,
                   s=55, alpha=0.75, edgecolors="white", linewidths=0.4)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(handles=legend_items, fontsize=9, loc="best", framealpha=0.9)

    plt.suptitle("PCA of Morgan Fingerprints (ECFP4, 1024 bits)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── Figure 5: SHAP feature importance ──────────────────────────────────────

def plot_shap_importance(model, X_train, X_test, feature_names: list,
                          save_path: str = "figures/fig5_shap_importance.png"):
    """SHAP beeswarm + bar for Random Forest model."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Bar plot (mean |SHAP|)
        mean_shap = np.abs(shap_vals).mean(axis=0)
        sorted_idx = np.argsort(mean_shap)[::-1][:15]
        axes[0].barh(
            [feature_names[i] for i in sorted_idx[::-1]],
            mean_shap[sorted_idx[::-1]],
            color="#1B7F79", alpha=0.85, edgecolor="white"
        )
        axes[0].set_xlabel("Mean |SHAP value|", fontsize=11)
        axes[0].set_title("Top 15 Features — Mean |SHAP|", fontsize=12, fontweight="bold")

        # Scatter: top feature
        top_feat = sorted_idx[0]
        feat_vals = X_test[:, top_feat]
        sc = axes[1].scatter(shap_vals[:, top_feat], feat_vals,
                             c=shap_vals[:, top_feat], cmap="RdBu_r",
                             s=60, alpha=0.8, edgecolors="white", linewidths=0.4)
        plt.colorbar(sc, ax=axes[1], label="SHAP value")
        axes[1].set_xlabel("SHAP value", fontsize=11)
        axes[1].set_ylabel(feature_names[top_feat], fontsize=11)
        axes[1].set_title(f"SHAP — {feature_names[top_feat]}", fontsize=12, fontweight="bold")
        axes[1].axvline(0, color="gray", lw=0.8, ls="--")

        plt.suptitle("SHAP Explainability — Random Forest", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {save_path}")
    except Exception as e:
        print(f"SHAP plot skipped: {e}")


# ── Figure 6: Descriptor distributions ──────────────────────────────────────

def plot_descriptor_distributions(df: pd.DataFrame,
                                   save_path: str = "figures/fig6_descriptor_distributions.png"):
    """Violin plots of key physicochemical properties by BBB class."""
    props = ["MolWt", "LogP", "TPSA", "NumHDonors",
             "NumHAcceptors", "NumRotatableBonds", "QED"]
    props = [p for p in props if p in df.columns]

    ncols = 4
    nrows = (len(props) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows))
    axes = axes.flatten()

    colors = {0: "#E74C3C", 1: "#2ECC71"}
    labels_map = {0: "BBB−", 1: "BBB+"}
    for i, prop in enumerate(props):
        ax = axes[i]
        for label, color in colors.items():
            subset = df[df["p_np"] == label][prop].dropna()
            parts = ax.violinplot([subset], positions=[label],
                                  showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.65)
            parts["cmedians"].set_color("black")
            ax.scatter([label] * len(subset),
                       subset, s=12, alpha=0.45,
                       color=color, edgecolors="none", zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["BBB−", "BBB+"], fontsize=10)
        ax.set_title(prop, fontsize=11, fontweight="bold")
        ax.set_ylabel("Value", fontsize=9)

    # Lipinski thresholds
    thresholds = {
        "MolWt": 500, "LogP": 5, "NumHDonors": 5,
        "NumHAcceptors": 10, "NumRotatableBonds": 10, "TPSA": 90,
    }
    for i, prop in enumerate(props):
        if prop in thresholds:
            axes[i].axhline(thresholds[prop], color="navy",
                            lw=1.2, ls="--", alpha=0.5,
                            label=f"Ro5: {thresholds[prop]}")
            axes[i].legend(fontsize=7.5, loc="upper right")

    for j in range(len(props), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Physicochemical Property Distributions by BBB Class",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── Results table ────────────────────────────────────────────────────────────

def print_results_table(all_metrics: dict):
    """Pretty-print a comparison table."""
    print("\n" + "=" * 80)
    print(f"{'Model':<25} {'AUC-ROC':>8} {'AUC-PR':>8} {'Accuracy':>9} {'F1':>7} {'Precision':>10} {'Recall':>8}")
    print("-" * 80)
    for model, m in all_metrics.items():
        print(f"{model:<25} {m.get('AUC-ROC',0):>8.4f} {m.get('AUC-PR',0):>8.4f} "
              f"{m.get('Accuracy',0):>9.4f} {m.get('F1',0):>7.4f} "
              f"{m.get('Precision',0):>10.4f} {m.get('Recall',0):>8.4f}")
    print("=" * 80 + "\n")
