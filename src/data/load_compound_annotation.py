import pandas as pd

def load_compound_annotation(path="data/raw/gdsc2/screened_compounds_rel_8.5.csv"):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())
    return df

if __name__ == "__main__":
    load_compound_annotation()