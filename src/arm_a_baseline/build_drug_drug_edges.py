import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def build_drug_drug_edges(drug_map_path="data/processed/drug_id_map_deduped.csv",
                           similarity_threshold=0.4,
                           out_path="data/processed/graph_drug_drug_edges.csv"):
    drug_map = pd.read_csv(drug_map_path)
    print(f"Drugs to compare: {len(drug_map)}")

    fps = {}
    for _, row in drug_map.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is not None:
            fps[row["DRUG_ID"]] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    print(f"Valid fingerprints: {len(fps)} / {len(drug_map)}")

    drug_ids = list(fps.keys())
    edges = []
    for i in range(len(drug_ids)):
        for j in range(i + 1, len(drug_ids)):
            sim = DataStructs.TanimotoSimilarity(fps[drug_ids[i]], fps[drug_ids[j]])
            if sim >= similarity_threshold:
                edges.append({"drug_a": drug_ids[i], "drug_b": drug_ids[j], "similarity": sim})

    edges_df = pd.DataFrame(edges)
    print(f"Edges above threshold {similarity_threshold}: {len(edges_df)}")
    os.makedirs("data/processed", exist_ok=True)
    edges_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return edges_df

if __name__ == "__main__":
    build_drug_drug_edges()