import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

from src.arm_a_baseline.features import build_drug_feature_table, load_expression_features
from src.arm_a_baseline.model import BaselineMLP

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


def load_predictions():
    return pd.read_csv("results/tables/arm_a_val_predictions.csv")


def plot_bootstrap_ci(n_bootstrap=1000, seed=42):
    df = load_predictions()
    rng = np.random.default_rng(seed)
    n = len(df)
    boot_spearmans = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        s = spearmanr(df["y_true"].values[idx], df["y_pred"].values[idx]).correlation
        boot_spearmans.append(s)
    boot_spearmans = np.array(boot_spearmans)
    lower, upper = np.percentile(boot_spearmans, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(boot_spearmans, bins=40, color="#7F77DD", alpha=0.8)
    ax.axvline(lower, color="#E8836B", linestyle="--", label=f"95% CI: [{lower:.4f}, {upper:.4f}]")
    ax.axvline(upper, color="#E8836B", linestyle="--")
    ax.axvline(boot_spearmans.mean(), color="white", linewidth=2)
    ax.set_xlabel("Bootstrapped Spearman correlation")
    ax.set_ylabel("Count")
    ax.set_title(f"Bootstrap distribution of val Spearman ({n_bootstrap} resamples)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "19_bootstrap_ci.png")
    plt.close()
    print(f"Saved 19_bootstrap_ci.png | 95% CI: [{lower:.4f}, {upper:.4f}]")


def plot_per_drug_ranking(min_pairs=30, top_n=15):
    df = load_predictions()
    drug_map = pd.read_csv("data/processed/drug_id_map_deduped.csv")[["DRUG_ID", "DRUG_NAME"]]
    drug_map = drug_map.rename(columns={"DRUG_ID": "drug_id"})
    results = []
    for drug_id, group in df.groupby("drug_id"):
        if len(group) < min_pairs:
            continue
        corr = spearmanr(group["y_true"], group["y_pred"]).correlation
        results.append({"drug_id": drug_id, "n_pairs": len(group), "spearman": corr})
    results_df = pd.DataFrame(results).merge(drug_map, on="drug_id", how="left")
    results_df["label"] = results_df["DRUG_NAME"].fillna(results_df["drug_id"].astype(str))
    results_df = results_df.sort_values("spearman", ascending=False)
    results_df.to_csv("results/tables/arm_a_per_drug.csv", index=False)

    top = results_df.head(top_n)
    bottom = results_df.tail(top_n)
    combined = pd.concat([top, bottom])

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#5FBF8F"] * len(top) + ["#E8836B"] * len(bottom)
    ax.barh(combined["label"], combined["spearman"], color=colors)
    ax.set_xlabel("Spearman correlation")
    ax.set_title(f"Best {top_n} and worst {top_n} predicted drugs")
    ax.axvline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "20_per_drug_ranking.png")
    plt.close()
    print("Saved 20_per_drug_ranking.png and results/tables/arm_a_per_drug.csv")


def plot_learned_embeddings(n_bits=1024, seed=42,
                             expr_path="data/raw/ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"):
    from sklearn.decomposition import PCA

    pairs_df = pd.read_csv("data/processed/training_table.csv")[["drug_id", "cell_line_id", "smiles", "label"]]
    drug_smiles_map = dict(zip(pairs_df["drug_id"], pairs_df["smiles"]))
    drug_features = build_drug_feature_table(drug_smiles_map, n_bits=n_bits)
    expr_features = load_expression_features(expr_path, pairs_df["cell_line_id"].unique().tolist())

    val_preds = load_predictions()
    valid = val_preds[
        val_preds["drug_id"].isin(drug_features.index) & val_preds["cell_line_id"].isin(expr_features.index)
    ].reset_index(drop=True)
    # subsample for speed if large
    if len(valid) > 5000:
        valid = valid.sample(5000, random_state=seed).reset_index(drop=True)

    input_dim = n_bits + expr_features.shape[1]
    model = BaselineMLP(input_dim=input_dim)
    model.load_state_dict(torch.load("results/arm_a_baseline_model.pt"))
    model.eval()

    X = np.stack([
        np.concatenate([drug_features.loc[row.drug_id].values, expr_features.loc[row.cell_line_id].values])
        for row in valid.itertuples()
    ]).astype(np.float32)

    with torch.no_grad():
        embeddings = model.net[:-1](torch.tensor(X)).numpy()  # everything up to (not including) the final output layer

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    errors = (valid["y_pred"] - valid["y_true"]).abs()

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=errors, cmap="magma", s=12, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="Absolute prediction error")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Model's learned representation (colored by prediction error)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "21_learned_embeddings.png")
    plt.close()
    print("Saved 21_learned_embeddings.png")


def plot_calibration_curve(n_bins=10):
    df = load_predictions()
    df = df.copy()
    df["bin"] = pd.qcut(df["y_pred"], n_bins, duplicates="drop")
    grouped = df.groupby("bin").agg(mean_pred=("y_pred", "mean"), mean_true=("y_true", "mean")).reset_index()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(grouped["mean_pred"], grouped["mean_true"], marker="o", color="#7F77DD", label="Model calibration")
    lims = [df["y_pred"].min(), df["y_pred"].max()]
    ax.plot(lims, lims, color="#E8836B", linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted response (per bin)")
    ax.set_ylabel("Mean actual response (per bin)")
    ax.set_title(f"Calibration curve ({n_bins} bins)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "22_calibration_curve.png")
    plt.close()
    print("Saved 22_calibration_curve.png")


if __name__ == "__main__":
    plot_bootstrap_ci()
    plot_per_drug_ranking()
    plot_learned_embeddings()
    plot_calibration_curve()