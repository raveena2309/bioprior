import pandas as pd

def check_overlap():
    # BioSNAP gene IDs
    biosnap = pd.read_csv("data/raw/biosnap/PP-Pathways_ppi.csv", header=None, names=["gene_a", "gene_b"])
    biosnap_genes = set(biosnap["gene_a"]).union(set(biosnap["gene_b"]))
    print(f"BioSNAP unique gene IDs: {len(biosnap_genes)}")
    print(f"Sample: {list(biosnap_genes)[:5]}")

    # Reactome gene IDs (first column of NCBI2Reactome.txt)
    reactome = pd.read_csv(
        "data/raw/reactome/NCBI2Reactome.txt", sep="\t", header=None,
        names=["gene_id", "reactome_id", "url", "pathway_name", "evidence", "species"]
    )
    reactome_genes = set(reactome["gene_id"])
    print(f"\nReactome unique gene IDs: {len(reactome_genes)}")
    print(f"Sample: {list(reactome_genes)[:5]}")

    # CCLE expression gene IDs (extracted from column names like "TSPAN6 (7105)")
    ccle_cols = pd.read_csv("data/raw/ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", nrows=0).columns
    ccle_genes = set()
    for col in ccle_cols:
        if "(" in col and ")" in col:
            try:
                gene_id = int(col.split("(")[1].strip(")"))
                ccle_genes.add(gene_id)
            except ValueError:
                pass
    print(f"\nCCLE unique gene IDs: {len(ccle_genes)}")
    print(f"Sample: {list(ccle_genes)[:5]}")

    # overlap checks
    biosnap_reactome = biosnap_genes.intersection(reactome_genes)
    biosnap_ccle = biosnap_genes.intersection(ccle_genes)
    reactome_ccle = reactome_genes.intersection(ccle_genes)
    all_three = biosnap_genes.intersection(reactome_genes).intersection(ccle_genes)

    print(f"\nBioSNAP ∩ Reactome: {len(biosnap_reactome)}")
    print(f"BioSNAP ∩ CCLE: {len(biosnap_ccle)}")
    print(f"Reactome ∩ CCLE: {len(reactome_ccle)}")
    print(f"All three: {len(all_three)}")

if __name__ == "__main__":
    check_overlap()