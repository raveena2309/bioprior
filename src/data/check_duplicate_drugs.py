import pandas as pd

edges = pd.read_csv("data/processed/graph_drug_drug_edges.csv")
drugs = pd.read_csv("data/processed/drug_id_map.csv")

exact = edges[edges["similarity"] == 1.0]
for _, row in exact.iterrows():
    name_a = drugs[drugs["DRUG_ID"] == row["drug_a"]]["DRUG_NAME"].values
    name_b = drugs[drugs["DRUG_ID"] == row["drug_b"]]["DRUG_NAME"].values
    a = name_a[0] if len(name_a) else "?"
    b = name_b[0] if len(name_b) else "?"
    print(f"{row['drug_a']} ({a})  --  {row['drug_b']} ({b})")