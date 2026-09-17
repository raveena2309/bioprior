import sqlite3
import pandas as pd
import os

CHEMBL_DB = "data/raw/chembl/chembl_37/chembl_37_sqlite/chembl_37.db"


def get_drug_target_edges(chembl_ids: list):
    conn = sqlite3.connect(CHEMBL_DB)
    placeholders = ",".join(["?"] * len(chembl_ids))
    query = f"""
    SELECT DISTINCT
        md.chembl_id,
        cs.accession AS uniprot_id,
        cs2.component_synonym AS gene_symbol
    FROM molecule_dictionary md
    JOIN drug_mechanism dm ON md.molregno = dm.molregno
    JOIN target_dictionary td ON dm.tid = td.tid
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences cs ON tc.component_id = cs.component_id
    JOIN component_synonyms cs2 ON tc.component_id = cs2.component_id
    WHERE md.chembl_id IN ({placeholders})
    AND cs2.syn_type = 'GENE_SYMBOL'
    """
    df = pd.read_sql_query(query, conn, params=chembl_ids)
    conn.close()
    return df


def build_drug_gene_edges(
    drug_map_path="data/processed/drug_id_map_deduped.csv",  # changed
    out_path="data/processed/graph_drug_gene_edges.csv",
):
    drugs = pd.read_csv(drug_map_path)
    chembl_ids = drugs["chembl_id"].dropna().unique().tolist()
    print(f"Querying targets for {len(chembl_ids)} drugs...")

    edges = get_drug_target_edges(chembl_ids)
    print(f"Raw drug-gene edges: {len(edges)}")
    print(f"Drugs with at least one target: {edges['chembl_id'].nunique()} / {len(chembl_ids)}")

    os.makedirs("data/processed", exist_ok=True)
    edges.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return edges


if __name__ == "__main__":
    build_drug_gene_edges()