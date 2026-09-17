import pandas as pd
import os

def build_gene_gene_edges(
    biosnap_path="data/raw/biosnap/PP-Pathways_ppi.csv",
    valid_genes: set = None,
    out_path="data/processed/graph_gene_gene_edges.csv",
):
    edges = pd.read_csv(biosnap_path, header=None, names=["gene_a", "gene_b"])
    print(f"Raw BioSNAP edges: {len(edges)}")

    if valid_genes is not None:
        edges = edges[edges["gene_a"].isin(valid_genes) & edges["gene_b"].isin(valid_genes)]
        print(f"Edges after filtering to shared gene universe: {len(edges)}")

    os.makedirs("data/processed", exist_ok=True)
    edges.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return edges

if __name__ == "__main__":
    build_gene_gene_edges()