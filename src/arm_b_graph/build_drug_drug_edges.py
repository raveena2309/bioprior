import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
import os

RDLogger.DisableLog('rdApp.*')


def build_drug_drug_edges(
    drug_map_path="data/processed/drug_id_map_deduped.csv",  # changed
    similarity_threshold: float = 0.7,
    out_path="data/processed/graph_drug_drug_edges.csv",
):
    drugs = pd.read_csv(drug_map_path)
    print(f"Drugs to compare: {len(drugs)}")

    # compute a fingerprint for each drug once
    fingerprints = {}
    for _, row in drugs.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is not None:
            fingerprints[row["DRUG_ID"]] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

    print(f"Valid fingerprints: {len(fingerprints)}")

    DRUG_IDs = list(fingerprints.keys())
    edges = []
    for i in range(len(DRUG_IDs)):
        for j in range(i + 1, len(DRUG_IDs)):
            sim = DataStructs.TanimotoSimilarity(fingerprints[DRUG_IDs[i]], fingerprints[DRUG_IDs[j]])
            if sim >= similarity_threshold:
                edges.append({"drug_a": DRUG_IDs[i], "drug_b": DRUG_IDs[j], "similarity": sim})

    result = pd.DataFrame(edges)
    print(f"Drug-drug edges above threshold {similarity_threshold}: {len(result)}")

    os.makedirs("data/processed", exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return result


if __name__ == "__main__":
    build_drug_drug_edges()