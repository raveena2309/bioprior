import pandas as pd
from src.arm_a_baseline.features import build_drug_feature_table, load_expression_features

def diagnose():
    pairs_df = pd.read_csv("data/processed/training_table.csv")[["drug_id", "cell_line_id", "smiles", "label"]]
    print(f"Total pairs: {len(pairs_df)}")

    drug_smiles_map = dict(zip(pairs_df["drug_id"], pairs_df["smiles"]))
    drug_features = build_drug_feature_table(drug_smiles_map, n_bits=1024)
    expr_features = load_expression_features(
        "data/raw/ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
        pairs_df["cell_line_id"].unique().tolist())

    missing_drug = ~pairs_df["drug_id"].isin(drug_features.index)
    missing_expr = ~pairs_df["cell_line_id"].isin(expr_features.index)

    print(f"\nRows missing drug features: {missing_drug.sum()}")
    print(f"Rows missing expression features: {missing_expr.sum()}")
    print(f"Rows missing BOTH: {(missing_drug & missing_expr).sum()}")
    print(f"\nUnique drugs affected: {pairs_df.loc[missing_drug, 'drug_id'].nunique()} of {pairs_df['drug_id'].nunique()}")
    print(f"Unique cell lines affected: {pairs_df.loc[missing_expr, 'cell_line_id'].nunique()} of {pairs_df['cell_line_id'].nunique()}")

    missing_smiles = pairs_df.loc[missing_drug, "smiles"].isna().sum()
    print(f"\nOf missing-drug rows: {missing_smiles} have no SMILES at all (expected drop), "
          f"{missing_drug.sum() - missing_smiles} have a SMILES that failed to parse — worth checking those manually")

if __name__ == "__main__":
    diagnose()