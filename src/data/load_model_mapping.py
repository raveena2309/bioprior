import pandas as pd
from pathlib import Path

def load_model_mapping(folder="data/raw/model_mapping"):
    folder = Path(folder)
    files = list(folder.glob("*.csv")) + list(folder.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No CSV/TSV found in {folder} — did the download actually save there?")
    path = files[0]
    print(f"Loading: {path}")
    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    sidm_cols = [c for c in df.columns if "sanger" in c.lower() or "sidm" in c.lower() or "model_id" in c.lower()]
    ach_cols = [c for c in df.columns if "broad" in c.lower() or "depmap" in c.lower() or "ach" in c.lower()]
    print("\nLikely SIDM column(s):", sidm_cols)
    print("Likely ACH column(s):", ach_cols)
    return df

if __name__ == "__main__":
    load_model_mapping()