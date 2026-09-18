import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import spearmanr

from src.arm_a_baseline.features import build_drug_feature_table, load_expression_features
from src.arm_a_baseline.dataset import DrugResponseBaselineDataset
from src.arm_a_baseline.model import BaselineMLP


def train_eval(train_ds, val_ds, input_dim, epochs, seed):
    torch.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    model = BaselineMLP(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for x, y in val_loader:
            preds.append(model(x).numpy())
            true.append(y.numpy())
    preds, true = np.concatenate(preds), np.concatenate(true)
    return float(np.mean((preds - true) ** 2)), float(spearmanr(preds, true).correlation)


def run_low_data_curve(fractions=(0.1, 0.25, 0.5, 1.0), epochs=30, seed=42,
                        expr_path="data/raw/ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"):
    pairs_df = pd.read_csv("data/processed/training_table.csv")[["drug_id", "cell_line_id", "smiles", "label"]]
    drug_smiles_map = dict(zip(pairs_df["drug_id"], pairs_df["smiles"]))
    drug_features = build_drug_feature_table(drug_smiles_map, n_bits=1024)
    expr_features = load_expression_features(expr_path, pairs_df["cell_line_id"].unique().tolist())

    full_dataset = DrugResponseBaselineDataset(pairs_df, drug_features, expr_features)
    input_dim = 1024 + expr_features.shape[1]

    # fixed val set (15%), held constant across every fraction so comparisons are apples-to-apples
    n_val = int(0.15 * len(full_dataset))
    n_pool = len(full_dataset) - n_val
    trainpool_ds, val_ds = random_split(full_dataset, [n_pool, n_val], generator=torch.Generator().manual_seed(seed))

    results = []
    rng = np.random.default_rng(seed)
    for frac in fractions:
        n_sub = int(len(trainpool_ds) * frac)
        sub_idx = rng.choice(len(trainpool_ds), size=n_sub, replace=False)
        train_subset = Subset(trainpool_ds, sub_idx)
        mse, spearman = train_eval(train_subset, val_ds, input_dim, epochs, seed)
        print(f"Fraction={frac:.0%} | n_train={n_sub} | val_mse={mse:.4f} | val_spearman={spearman:.4f}")
        results.append({"fraction": frac, "n_train": n_sub, "val_mse": mse, "val_spearman": spearman})

    with open("results/tables/arm_a_low_data_curve.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results/tables/arm_a_low_data_curve.json")
    return results


if __name__ == "__main__":
    run_low_data_curve()