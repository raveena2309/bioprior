import os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from scipy.stats import spearmanr

from src.arm_a_baseline.features import build_drug_feature_table, load_expression_features
from src.arm_a_baseline.dataset import DrugResponseBaselineDataset
from src.arm_a_baseline.model import BaselineMLP


def load_training_pairs():
    df = pd.read_csv("data/processed/training_table.csv")
    return df[["drug_id", "cell_line_id", "smiles", "label"]]


def train_and_save(expr_path="data/raw/ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
                    n_bits=1024, epochs=50, batch_size=64, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    pairs_df = load_training_pairs()
    drug_smiles_map = dict(zip(pairs_df["drug_id"], pairs_df["smiles"]))
    drug_features = build_drug_feature_table(drug_smiles_map, n_bits=n_bits)
    expr_features = load_expression_features(expr_path, pairs_df["cell_line_id"].unique().tolist())

    valid_mask = pairs_df["drug_id"].isin(drug_features.index) & pairs_df["cell_line_id"].isin(expr_features.index)
    filtered_pairs = pairs_df[valid_mask].reset_index(drop=True)
    dataset = DrugResponseBaselineDataset(pairs_df, drug_features, expr_features)
    
    n_val = int(0.15 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)  # order preserved, no shuffle

    input_dim = n_bits + expr_features.shape[1]
    model = BaselineMLP(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    history = []
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
        val_mse = float(np.mean((val_preds - val_true) ** 2))
        val_spearman = float(spearmanr(val_preds, val_true).correlation)

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_mse": val_mse, "val_spearman": val_spearman})
        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_mse={val_mse:.4f} | val_spearman={val_spearman:.4f}")

    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/arm_a_training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Saved results/tables/arm_a_training_history.json")

    # save labeled val predictions (drug_id + cell_line_id attached) for scatter plot + per-cancer-type analysis
    val_meta = filtered_pairs.iloc[val_ds.indices].reset_index(drop=True)
    val_meta["y_true"] = val_true
    val_meta["y_pred"] = val_preds
    val_meta.to_csv("results/tables/arm_a_val_predictions.csv", index=False)
    print("Saved results/tables/arm_a_val_predictions.csv")

    torch.save(model.state_dict(), "results/arm_a_baseline_model.pt")
    return model, history


if __name__ == "__main__":
    train_and_save()