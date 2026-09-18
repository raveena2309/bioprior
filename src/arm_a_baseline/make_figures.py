"""
Generates presentation-ready figures from saved results.
Run after arm_a_final_submission.json, kfold/unseen results, and
drug_id_map files all exist.
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


def fig1_evaluation_comparison():
    with open("results/tables/arm_a_final_submission.json") as f:
        summary = json.load(f)

    labels = ["Single split\n(easiest)", "5-fold CV\n(more rigorous)", "Unseen drug\n(hardest)"]
    values = [
        summary["single_split_result"]["val_spearman"],
        summary["kfold_result"]["mean_val_spearman"],
        summary["unseen_drug_result"]["test_spearman"],
    ]
    colors = ["#8FA9DB", "#7F77DD", "#E8836B"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Arm A (Baseline) — performance across evaluation difficulty")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "1_evaluation_comparison.png")
    plt.close()
    print("Saved 1_evaluation_comparison.png")


def fig2_kfold_spread():
    with open("results/tables/arm_a_kfold_result.json") as f:
        kfold = json.load(f)

    folds = [f["fold"] for f in kfold["folds"]]
    scores = [f["val_spearman"] for f in kfold["folds"]]
    mean = kfold["mean_val_spearman"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(folds, scores, s=80, color="#7F77DD", zorder=3, label="Individual fold")
    ax.axhline(mean, color="#E8836B", linestyle="--", label=f"Mean = {mean:.4f}")
    ax.set_xticks(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Val Spearman correlation")
    ax.set_title("5-fold cross-validation — consistency across folds")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "2_kfold_spread.png")
    plt.close()
    print("Saved 2_kfold_spread.png")


def fig3_drug_matching_funnel(gdsc2_total_drugs=295, matched_drugs=172, final_drugs=172):
    stages = ["GDSC2\nraw drugs", "Matched to\nChEMBL", "In final\ntraining table"]
    counts = [gdsc2_total_drugs, matched_drugs, final_drugs]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(stages, counts, color=["#8FA9DB", "#7F77DD", "#5FBF8F"])
    ax.set_ylabel("Number of drugs")
    ax.set_title("Drug ID matching funnel")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, val + 3, str(val), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "3_drug_matching_funnel.png")
    plt.close()
    print("Saved 3_drug_matching_funnel.png")


def fig4_dataset_sizes():
    data = {
        "GDSC2\n(242K rows)": 242036,
        "BioSNAP\n(edges)": 342353,
        "Reactome\ngene-pathway": 49642,
        "Cell-line\nmap": 1770,
        "Training\ntable": 151973,
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(data.keys(), data.values(), color="#7F77DD")
    ax.set_ylabel("Row count")
    ax.set_title("Dataset sizes across the pipeline")
    ax.set_yscale("log")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "4_dataset_sizes.png")
    plt.close()
    print("Saved 4_dataset_sizes.png (log scale — sizes vary by orders of magnitude)")


def fig5_gene_overlap(biosnap_ccle=16431, three_way=9307):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["BioSNAP ∩ CCLE", "BioSNAP ∩ Reactome\n∩ CCLE (used)"]
    values = [biosnap_ccle, three_way]
    bars = ax.bar(labels, values, color=["#8FA9DB", "#5FBF8F"])
    ax.set_ylabel("Number of genes")
    ax.set_title("Gene ID overlap across sources")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 200, str(val), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "5_gene_overlap.png")
    plt.close()
    print("Saved 5_gene_overlap.png")


def fig6_graph_edge_counts():
    edges = {
        "Gene-gene\n(BioSNAP)": 342353,
        "Gene-pathway\n(Reactome)": 49642,
        "Pathway\nhierarchy": 2899,
        "Drug-gene\n(target)": 347,
        "Drug-drug\n(similarity)": 5,
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(edges.keys(), edges.values(), color="#7F77DD")
    ax.set_ylabel("Edge count (log scale)")
    ax.set_yscale("log")
    ax.set_title("Arm B knowledge graph — edge types built so far")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "6_graph_edge_counts.png")
    plt.close()
    print("Saved 6_graph_edge_counts.png")


if __name__ == "__main__":
    fig1_evaluation_comparison()
    fig2_kfold_spread()
    fig3_drug_matching_funnel()
    fig4_dataset_sizes()
    fig5_gene_overlap()
    fig6_graph_edge_counts()
    print(f"\nAll figures saved to {FIG_DIR}/")