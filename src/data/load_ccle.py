import pandas as pd

def load_ccle_mutations(path="data/raw/ccle/OmicsSomaticMutations.csv"):
    df = pd.read_csv(path, nrows=1000)  # just a peek, full file is 580MB
    print("Mutations shape (sample):", df.shape)
    print("Mutations columns:", list(df.columns))
    print(df.head())
    return df

def load_ccle_expression(path="data/raw/ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"):
    df = pd.read_csv(path, nrows=5)  # just a peek, full file is 305MB
    print("\nExpression shape (sample):", df.shape)
    print("Expression columns (first 10):", list(df.columns[:10]))
    print(df.iloc[:, :5])
    return df

if __name__ == "__main__":
    load_ccle_mutations()
    load_ccle_expression()