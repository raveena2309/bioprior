import pandas as pd
import os

def build_gene_pathway_edges(
    reactome_path="data/raw/reactome/NCBI2Reactome.txt",
    out_path="data/processed/graph_gene_pathway_edges.csv",
):
    df = pd.read_csv(
        reactome_path, sep="\t", header=None,
        names=["gene_id", "reactome_id", "url", "pathway_name", "evidence", "species"],
        low_memory=False,
    )
    print(f"Raw Reactome rows: {len(df)}")

    # filter to human only
    df = df[df["species"] == "Homo sapiens"]
    print(f"After filtering to Homo sapiens: {len(df)}")

    # drop rows where gene_id isn't actually numeric (the source of that mixed-type warning)
    df["gene_id"] = pd.to_numeric(df["gene_id"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["gene_id"])
    df["gene_id"] = df["gene_id"].astype(int)
    print(f"Dropped {before - len(df)} rows with non-numeric gene_id")

    result = df[["gene_id", "reactome_id", "pathway_name"]].drop_duplicates()
    print(f"Final gene-pathway edges: {len(result)}")
    print(f"Unique genes: {result['gene_id'].nunique()}")
    print(f"Unique pathways: {result['reactome_id'].nunique()}")

    os.makedirs("data/processed", exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return result

if __name__ == "__main__":
    build_gene_pathway_edges()