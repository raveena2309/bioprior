import pandas as pd
import os

def build_pathway_hierarchy(
    relation_path="data/raw/reactome/ReactomePathwaysRelation.txt",
    pathways_path="data/raw/reactome/ReactomePathways.txt",
    out_path="data/processed/graph_pathway_hierarchy_edges.csv",
):
    # get ALL human pathway IDs (not just ones with direct gene annotations)
    pathways = pd.read_csv(
        pathways_path, sep="\t", header=None,
        names=["pathway_id", "pathway_name", "species"]
    )
    human_pathways = set(pathways[pathways["species"] == "Homo sapiens"]["pathway_id"])
    print(f"Human pathway IDs (from ReactomePathways.txt): {len(human_pathways)}")

    df = pd.read_csv(relation_path, sep="\t", header=None, names=["parent_pathway", "child_pathway"])
    print(f"Raw pathway relation edges: {len(df)}")

    df = df[df["parent_pathway"].isin(human_pathways) & df["child_pathway"].isin(human_pathways)]
    print(f"After filtering to human pathways: {len(df)}")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return df


if __name__ == "__main__":
    build_pathway_hierarchy()