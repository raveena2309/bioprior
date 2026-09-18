"""
Unseen-drug generalization test: hold out entire drugs (not just pairs),
so the model is tested on drugs it has never seen in any form during training.
This is the test that actually shows whether the model learned something
generalizable, vs. memorizing per-drug behavior.
"""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from src.arm_a_baseline.features import build_drug_feature_table, load_expression_features
from src.arm_a_baseline.dataset import DrugResponseBaselineDataset
from src.arm_a_baseline.model import BaselineMLP


def run_unseen_drug_eval(test_fraction=0.2, n_bits=1024, epochs=40, seed=42,
                          expr_path="data/raw/ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"):
    torch.manual_seed(seed)
    pairs_df = pd.read_csv("data/processed/training_table.csv")[["drug_id", "cell_line_id", "smiles", "label"]]

    unique_drugs = pairs_df["drug_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_drugs)
    n_test_drugs = int(len(unique_drugs) * test_fraction)
    test_drugs = set(unique_drugs[:n_test_drugs])
    train_drugs = set(unique_drugs[n_test_drugs:])

    train_df = pairs_df[pairs_df["drug_id"].isin(train_drugs)].reset_index(drop=True)
    test_df = pairs_df[pairs_df["drug_id"].isin(test_drugs)].reset_index(drop=True)

    print(f"Train drugs: {len(train_drugs)} ({len(train_df)} pairs)")
    print(f"Test drugs (never seen in training): {len(test_drugs)} ({len(test_df)} pairs)")

    drug_smiles_map = dict(zip(pairs_df["drug_id"], pairs_df["smiles"]))
    drug_features = build_drug_feature_table(drug_smiles_map, n_bits=n_bits)
    expr_features = load_expression_features(expr_path, pairs_df["cell_line_id"].unique().tolist())

    train_ds = DrugResponseBaselineDataset(train_df, drug_features, expr_features)
    test_ds = DrugResponseBaselineDataset(test_df, drug_features, expr_features)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    input_dim = n_bits + expr_features.shape[1]
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
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} done")

    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x).numpy())
            true.append(y.numpy())
    preds = np.concatenate(preds)
    true = np.concatenate(true)
    mse = float(np.mean((preds - true) ** 2))
    spearman = float(spearmanr(preds, true).correlation)

    result = {
        "arm": "A_baseline",
        "evaluation": "unseen-drug generalization",
        "n_train_drugs": len(train_drugs),
        "n_test_drugs": len(test_drugs),
        "n_train_pairs": len(train_df),
        "n_test_pairs": len(test_df),
        "test_mse": mse,
        "test_spearman": spearman,
    }

    with open("results/tables/arm_a_unseen_drug_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nUnseen-drug test_spearman: {spearman:.4f} | test_mse: {mse:.4f}")
    print("Saved to results/tables/arm_a_unseen_drug_result.json")
    return result


if __name__ == "__main__":
    run_unseen_drug_eval()
    