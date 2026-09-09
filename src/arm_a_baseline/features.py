"""
Builds the two feature blocks the baseline model needs:
1. Drug fingerprints (Morgan/ECFP) from SMILES strings (ChEMBL)
2. Cell-line expression vectors (CCLE)

No biological prior here on purpose — this arm is the "raw features only" control.
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Convert one SMILES string to a Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def build_drug_feature_table(drug_smiles_map: dict, n_bits: int = 1024) -> pd.DataFrame:
    """
    drug_smiles_map: {drug_id: smiles_string}
    Returns a DataFrame indexed by drug_id, columns fp_0 ... fp_{n_bits-1}
    """
    rows = {
        drug_id: smiles_to_fingerprint(smiles, n_bits=n_bits)
        for drug_id, smiles in drug_smiles_map.items()
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.columns = [f"fp_{i}" for i in range(n_bits)]
    df.index.name = "drug_id"
    return df


def load_expression_features(expr_path: str, cell_line_ids: list) -> pd.DataFrame:
    expr = pd.read_csv(expr_path)
    # keep only the default entry per model — some models have multiple sequencing entries
    expr = expr[expr["IsDefaultEntryForModel"] == "Yes"]
    expr = expr.set_index("ModelID")
    # drop non-gene metadata columns, keep only gene expression columns
    meta_cols = ["Unnamed: 0", "SequencingID", "ModelConditionID", "IsDefaultEntryForMC", "IsDefaultEntryForModel"]
    expr = expr.drop(columns=[c for c in meta_cols if c in expr.columns])
    expr = expr.loc[expr.index.intersection(cell_line_ids)]
    expr = (expr - expr.mean(axis=0)) / (expr.std(axis=0) + 1e-8)
    return expr