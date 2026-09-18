"""
Proper k-fold cross-validation for Arm A, run on the CURRENT training_table.csv
(151,973 rows) — supersedes the earlier single-split result.
"""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from src.arm_a_baseline.features import build_drug_feature_table, load_expression_features
from src.arm_a_baseline.dataset import DrugResponseBaselineDataset
from src.arm_a_baseline.model import BaselineMLP


def load_training_pairs():
    df = pd.read_csv("data/processed/training_table.csv")
    return df[["drug_id", "cell_line_id", "smiles", "label"]]


def train_one_fold(train_ds, val_ds, input_dim, epochs=30, batch_size=64, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BaselineMLP(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for x, y in val_loader:
            val_preds.append(model(x).numpy())
            val_true.append(y.numpy())
    val_preds = np.concatenate(val_preds)
    val_true = np.concatenate(val_true)
    mse = float(np.mean((val_preds - val_true) ** 2))
    spearman = float(spearmanr(val_preds, val_true).correlation)
    return mse, spearman


def run_kfold(n_splits=5, n_bits=1024, epochs=30, seed=42,
              expr_path="data/raw/ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"):
    pairs_df = load_training_pairs()
    drug_smiles_map = dict(zip(pairs_df["drug_id"], pairs_df["smiles"]))
    drug_features = build_drug_feature_table(drug_smiles_map, n_bits=n_bits)
    expr_features = load_expression_features(expr_path, pairs_df["cell_line_id"].unique().tolist())

    full_dataset = DrugResponseBaselineDataset(pairs_df, drug_features, expr_features)
    input_dim = n_bits + expr_features.shape[1]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(full_dataset))

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_ds = Subset(full_dataset, train_idx)
        val_ds = Subset(full_dataset, val_idx)
        mse, spearman = train_one_fold(train_ds, val_ds, input_dim, epochs=epochs, seed=seed)
        print(f"Fold {fold+1}/{n_splits} | val_mse={mse:.4f} | val_spearman={spearman:.4f}")
        fold_results.append({"fold": fold + 1, "val_mse": mse, "val_spearman": spearman})

    spearmans = [r["val_spearman"] for r in fold_results]
    mses = [r["val_mse"] for r in fold_results]

    summary = {
        "arm": "A_baseline",
        "evaluation": "5-fold cross-validation",
        "training_table_rows": len(pairs_df),
        "n_splits": n_splits,
        "epochs_per_fold": epochs,
        "folds": fold_results,
        "mean_val_spearman": float(np.mean(spearmans)),
        "std_val_spearman": float(np.std(spearmans)),
        "mean_val_mse": float(np.mean(mses)),
        "std_val_mse": float(np.std(mses)),
    }

    with open("results/tables/arm_a_kfold_result.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean val_spearman: {summary['mean_val_spearman']:.4f} ± {summary['std_val_spearman']:.4f}")
    print("Saved to results/tables/arm_a_kfold_result.json")
    return summary


if __name__ == "__main__":
    run_kfold()