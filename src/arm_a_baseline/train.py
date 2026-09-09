"""
Training loop for the baseline arm.
"""
import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from scipy.stats import spearmanr
import numpy as np

from src.arm_a_baseline.features import build_drug_feature_table, load_expression_features
from src.arm_a_baseline.dataset import DrugResponseBaselineDataset
from src.arm_a_baseline.model import BaselineMLP


def load_training_pairs():
    df = pd.read_csv("data/processed/training_table.csv")
    return df[["drug_id", "cell_line_id", "smiles", "label"]]


def train_baseline(
    expr_path="data/raw/ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
    n_bits=1024,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    seed=42,
):
    torch.manual_seed(seed)

    pairs_df = load_training_pairs()
    drug_smiles_map = dict(zip(pairs_df["drug_id"], pairs_df["smiles"]))
    drug_features = build_drug_feature_table(drug_smiles_map, n_bits=n_bits)
    expr_features = load_expression_features(expr_path, pairs_df["cell_line_id"].unique().tolist())

    dataset = DrugResponseBaselineDataset(pairs_df, drug_features, expr_features)

    n_val = int(0.15 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    input_dim = n_bits + expr_features.shape[1]
    model = BaselineMLP(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                val_preds.append(model(x).numpy())
                val_true.append(y.numpy())
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_mse = np.mean((val_preds - val_true) ** 2)
        val_spearman = spearmanr(val_preds, val_true).correlation

        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_mse={val_mse:.4f} | val_spearman={val_spearman:.4f}")

    # --- save results, now that train_loss/val_mse/val_spearman/model are all in scope ---
    os.makedirs("results/tables", exist_ok=True)
    final_result = {
        "arm": "A_baseline",
        "final_train_loss": train_loss,
        "final_val_mse": float(val_mse),
        "final_val_spearman": float(val_spearman),
        "epochs": epochs,
        "n_train": n_train,
        "n_val": n_val,
    }
    with open("results/tables/arm_a_baseline_result.json", "w") as f:
        json.dump(final_result, f, indent=2)
    print(f"\nSaved result to results/tables/arm_a_baseline_result.json")

    torch.save(model.state_dict(), "results/arm_a_baseline_model.pt")
    print("Saved model weights to results/arm_a_baseline_model.pt")

    return model


if __name__ == "__main__":
    train_baseline()