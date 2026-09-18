import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


def plot_training_curves():
    with open("results/tables/arm_a_training_history.json") as f:
        df = pd.DataFrame(json.load(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train loss", color="#7F77DD")
    axes[0].plot(df["epoch"], df["val_mse"], label="Val MSE", color="#E8836B")
    axes[0].set_xlabel("Epoch"); axes[0].set_title("Training curve"); axes[0].legend()
    axes[1].plot(df["epoch"], df["val_spearman"], color="#5FBF8F")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Spearman"); axes[1].set_title("Validation Spearman over training")
    plt.tight_layout(); plt.savefig(FIG_DIR / "7_training_curves.png"); plt.close()
    print("Saved 7_training_curves.png")


def plot_predicted_vs_actual():
    df = pd.read_csv("results/tables/arm_a_val_predictions.csv")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["y_true"], df["y_pred"], alpha=0.15, s=10, color="#7F77DD")
    lims = [min(df["y_true"].min(), df["y_pred"].min()), max(df["y_true"].max(), df["y_pred"].max())]
    ax.plot(lims, lims, color="#E8836B", linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual response"); ax.set_ylabel("Predicted response")
    ax.set_title("Predicted vs. Actual — validation set"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "8_predicted_vs_actual.png"); plt.close()
    print("Saved 8_predicted_vs_actual.png")


def plot_low_data_curve():
    with open("results/tables/arm_a_low_data_curve.json") as f:
        df = pd.DataFrame(json.load(f))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(df["n_train"], df["val_spearman"], marker="o", color="#7F77DD")
    ax.set_xlabel("Training set size (pairs)"); ax.set_ylabel("Val Spearman correlation")
    ax.set_title("Low-data curve — does more data keep helping?")
    plt.tight_layout(); plt.savefig(FIG_DIR / "9_low_data_curve.png"); plt.close()
    print("Saved 9_low_data_curve.png")


def plot_per_cancer_type():
    df = pd.read_csv("results/tables/arm_a_per_cancer_type.csv")
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.3)))
    ax.barh(df["cancer_type"], df["spearman"], color="#7F77DD")
    ax.set_xlabel("Spearman correlation"); ax.set_title("Performance by cancer type")
    plt.tight_layout(); plt.savefig(FIG_DIR / "10_per_cancer_type.png"); plt.close()
    print("Saved 10_per_cancer_type.png")


if __name__ == "__main__":
    plot_training_curves()
    plot_predicted_vs_actual()
    plot_low_data_curve()
    plot_per_cancer_type()