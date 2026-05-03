
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_model_comparison(results: dict, save_path: str = "results/model_comparison.png"):
    """Bar chart comparing accuracy, F1, and ROC-AUC across models."""
    _ensure_dir(save_path)

    names   = list(results.keys())
    acc     = [results[n]["accuracy"]  for n in names]
    f1      = [results[n]["f1_macro"]  for n in names]
    auc     = [results[n]["roc_auc"] or 0 for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width, acc, width, label="Accuracy",  color="#4C72B0")
    b2 = ax.bar(x,          f1,  width, label="F1 Macro",  color="#55A868")
    b3 = ax.bar(x + width,  auc, width, label="ROC-AUC",   color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("Model Comparison — Fake News Detector", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved → {save_path}")


def plot_confusion_matrix(
    confusion: np.ndarray,
    model_name: str,
    save_path: str = "results/confusion_matrix.png",
):
    """Heatmap of the confusion matrix for the best model."""
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved → {save_path}")


def plot_top_features(
    pipeline,
    n: int = 20,
    save_path: str = "results/top_features.png",
):
    """
    Show the top n TF-IDF features (words/bigrams) by absolute
    coefficient weight for Logistic Regression models.
    Skipped for tree-based models.
    """
    _ensure_dir(save_path)

    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "coef_"):
        print(f"[plot]  Feature importance plot skipped (not supported for this model).")
        return

    tfidf     = pipeline.named_steps["tfidf"]
    feature_names = np.array(tfidf.get_feature_names_out())
    coef          = clf.coef_[0]

    top_pos_idx = np.argsort(coef)[-n:]
    top_neg_idx = np.argsort(coef)[:n]
    indices     = np.concatenate([top_neg_idx, top_pos_idx])

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#C44E52" if coef[i] > 0 else "#4C72B0" for i in indices]
    ax.barh(feature_names[indices], coef[indices], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient Weight", fontsize=11)
    ax.set_title("Top Features: Fake (red) vs Real (blue)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved → {save_path}")
