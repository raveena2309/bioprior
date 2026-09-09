"""
PyTorch Dataset for the baseline arm.
Wraps the joined (drug, cell_line) -> label table plus the two feature blocks.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class DrugResponseBaselineDataset(Dataset):
    def __init__(self, pairs_df: pd.DataFrame, drug_features: pd.DataFrame, expr_features: pd.DataFrame):
        # keep only pairs where both drug and cell-line features actually exist
        valid = pairs_df[
            pairs_df["drug_id"].isin(drug_features.index)
            & pairs_df["cell_line_id"].isin(expr_features.index)
        ].reset_index(drop=True)

        dropped = len(pairs_df) - len(valid)
        if dropped > 0:
            print(f"Dropped {dropped} pairs missing drug or expression features")

        # --- fast path: convert to numpy arrays + index dicts once, upfront ---
        self.drug_id_to_row = {drug_id: i for i, drug_id in enumerate(drug_features.index)}
        self.expr_id_to_row = {cl_id: i for i, cl_id in enumerate(expr_features.index)}

        self.drug_array = drug_features.values.astype(np.float32)
        self.expr_array = expr_features.values.astype(np.float32)

        self.drug_row_idx = valid["drug_id"].map(self.drug_id_to_row).values
        self.expr_row_idx = valid["cell_line_id"].map(self.expr_id_to_row).values
        self.labels = valid["label"].values.astype(np.float32)

        self.length = len(valid)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        drug_vec = self.drug_array[self.drug_row_idx[idx]]
        expr_vec = self.expr_array[self.expr_row_idx[idx]]
        x = torch.from_numpy(np.concatenate([drug_vec, expr_vec]))
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y