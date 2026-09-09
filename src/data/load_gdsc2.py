import pandas as pd

def load_gdsc2(path="data/raw/gdsc2/GDSC2_fitted_dose_response_27Oct23.xlsx"):
    df = pd.read_excel(path)
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nUnique drugs:", df['DRUG_NAME'].nunique() if 'DRUG_NAME' in df.columns else "column not found")
    print("Unique cell lines:", df['CELL_LINE_NAME'].nunique() if 'CELL_LINE_NAME' in df.columns else "column not found")
    return df

if __name__ == "__main__":
    load_gdsc2()