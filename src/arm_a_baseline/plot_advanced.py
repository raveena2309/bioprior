import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


def load_predictions():
    return pd.read_csv("results/tables/arm_a_val_predictions.csv")


def plot_hexbin():
    df = load_predictions()
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(df["y_true"], df["y_pred"], gridsize=40, cmap="viridis", mincnt=1)
    lims = [min(df["y_true"].min(), df["y_pred"].min()), max(df["y_true"].max(), df["y_pred"].max())]
    ax.plot(lims, lims, color="white", linestyle="--", linewidth=1.5, label="Perfect prediction")
    fig.colorbar(hb, ax=ax, label="Number of predictions")
    ax.set_xlabel("Actual response"); ax.set_ylabel("Predicted response")
    ax.set_title("Prediction density — Actual vs. Predicted")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "15_hexbin_density.png")
    plt.close()
    print("Saved 15_hexbin_density.png")


def plot_residuals():
    df = load_predictions()
    residuals = df["y_pred"] - df["y_true"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(df["y_true"], residuals, alpha=0.15, s=8, color="#7F77DD")
    axes[0].axhline(0, color="#E8836B", linestyle="--")
    axes[0].set_xlabel("Actual response"); axes[0].set_ylabel("Residual (Predicted - Actual)")
    axes[0].set_title("Residuals vs. Actual response")

    axes[1].hist(residuals, bins=60, color="#7F77DD", alpha=0.8)
    axes[1].axvline(0, color="#E8836B", linestyle="--")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residual distribution (mean={residuals.mean():.3f}, std={residuals.std():.3f})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "16_residuals.png")
    plt.close()
    print("Saved 16_residuals.png")


def plot_roc_and_confusion(threshold_percentile=50):
    """
    Reframes regression as classification: is a drug 'sensitive' (below median response)
    or 'resistant' (above)? Median split is a standard, defensible threshold choice.
    """
    df = load_predictions()
    threshold = np.percentile(df["y_true"], threshold_percentile)
    y_true_binary = (df["y_true"] < threshold).astype(int)  # 1 = sensitive
    y_score = -df["y_pred"]  # lower predicted response = more "sensitive" score

    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    auc = roc_auc_score(y_true_binary, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#7F77DD", linewidth=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC — Sensitive vs. Resistant classification")
    axes[0].legend()

    y_pred_binary = (df["y_pred"] < threshold).astype(int)
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Resistant", "Sensitive"])
    disp.plot(ax=axes[1], cmap="Purples", colorbar=False)
    axes[1].set_title("Confusion matrix (median-split threshold)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "17_roc_confusion.png")
    plt.close()
    print(f"Saved 17_roc_confusion.png (AUC = {auc:.3f})")


def plot_error_by_cancer_type():
    df = load_predictions()
    cellline_map = pd.read_csv("data/processed/cellline_id_map.csv")
    gdsc2 = pd.read_excel("data/raw/gdsc2/GDSC2_fitted_dose_response_27Oct23.xlsx")
    gdsc2_lookup = gdsc2[["SANGER_MODEL_ID", "CANCER_TYPE"]].drop_duplicates(subset=["SANGER_MODEL_ID"])

    merged = df.merge(cellline_map, left_on="cell_line_id", right_on="ccle_model_id", how="left")
    merged = merged.merge(gdsc2_lookup, left_on="sanger_model_id", right_on="SANGER_MODEL_ID", how="left")
    merged["abs_error"] = (merged["y_pred"] - merged["y_true"]).abs()

    top_cancers = merged["CANCER_TYPE"].value_counts().nlargest(12).index
    plot_df = merged[merged["CANCER_TYPE"].isin(top_cancers)]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.boxplot(column="abs_error", by="CANCER_TYPE", ax=ax, rot=45, grid=False,
                     patch_artist=True, boxprops=dict(facecolor="#7F77DD", alpha=0.6))
    ax.set_xlabel(""); ax.set_ylabel("Absolute prediction error")
    ax.set_title("Prediction error spread by cancer type (top 12 by sample count)")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "18_error_boxplot_by_cancer.png")
    plt.close()
    print("Saved 18_error_boxplot_by_cancer.png")


if __name__ == "__main__":
    plot_hexbin()
    plot_residuals()
    plot_roc_and_confusion()
    plot_error_by_cancer_type()