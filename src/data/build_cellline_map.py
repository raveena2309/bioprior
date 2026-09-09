import pandas as pd
import os

def build_cellline_map(
    model_list_path="data/raw/model_mapping/model_list_20260814.csv",
    out_path="data/processed/cellline_id_map.csv"
):
    df = pd.read_csv(model_list_path)

    # confirm BROAD_ID actually looks like ACH-xxxxxx
    sample_broad = df["BROAD_ID"].dropna().head(5).tolist()
    print("Sample BROAD_ID values:", sample_broad)

    mapping = df[["model_id", "BROAD_ID"]].dropna()
    mapping = mapping.rename(columns={"model_id": "sanger_model_id", "BROAD_ID": "ccle_model_id"})

    print(f"\nTotal models with both IDs: {len(mapping)}")

    os.makedirs("data/processed", exist_ok=True)
    mapping.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    return mapping

if __name__ == "__main__":
    build_cellline_map()