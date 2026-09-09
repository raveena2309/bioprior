import pandas as pd
import os

def build_training_table(
    gdsc2_path="data/raw/gdsc2/GDSC2_fitted_dose_response_27Oct23.xlsx",
    cellline_map_path="data/processed/cellline_id_map.csv",
    drug_map_path="data/processed/drug_id_map.csv",
    label_column="LN_IC50",
    out_path="data/processed/training_table.csv",
):
    gdsc2 = pd.read_excel(gdsc2_path)
    cellline_map = pd.read_csv(cellline_map_path)
    drug_map = pd.read_csv(drug_map_path)

    print("GDSC2 rows:", len(gdsc2))

    # join cell-line IDs: GDSC2's SANGER_MODEL_ID -> CCLE's ccle_model_id
    merged = gdsc2.merge(
        cellline_map, left_on="SANGER_MODEL_ID", right_on="sanger_model_id", how="inner"
    )
    print("After cell-line join:", len(merged))

    # join drug IDs: GDSC2's DRUG_ID -> chembl_id + smiles
    merged = merged.merge(
        drug_map[["DRUG_ID", "chembl_id", "canonical_smiles"]], on="DRUG_ID", how="inner"
    )
    print("After drug join:", len(merged))

    result = merged[[
        "DRUG_ID", "DRUG_NAME", "chembl_id", "canonical_smiles",
        "ccle_model_id", "CELL_LINE_NAME", "PUTATIVE_TARGET", "PATHWAY_NAME",
        label_column
    ]].rename(columns={
        "DRUG_ID": "drug_id",
        "canonical_smiles": "smiles",
        "ccle_model_id": "cell_line_id",
        label_column: "label",
    })

    result = result.dropna(subset=["smiles", "label"])
    print("Final training table:", len(result), "rows")
    print("Unique drugs:", result["drug_id"].nunique())
    print("Unique cell lines:", result["cell_line_id"].nunique())

    os.makedirs("data/processed", exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return result

if __name__ == "__main__":
    build_training_table()