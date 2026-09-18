"""
Pulls together all three Arm A results (original single-split, k-fold, unseen-drug)
into one clean file for submission.
"""
import json

def build_submission_summary():
    with open("results/tables/arm_a_baseline_result.json") as f:
        single_split = json.load(f)
    with open("results/tables/arm_a_kfold_result.json") as f:
        kfold = json.load(f)
    with open("results/tables/arm_a_unseen_drug_result.json") as f:
        unseen = json.load(f)

    summary = {
        "arm": "A - Baseline (no biological prior)",
        "single_split_result": {
            "note": "Original run, on an earlier 113,645-row training table",
            "val_spearman": single_split["final_val_spearman"],
            "val_mse": single_split["final_val_mse"],
        },
        "kfold_result": {
            "note": f"{kfold['n_splits']}-fold CV on current {kfold['training_table_rows']}-row table",
            "mean_val_spearman": kfold["mean_val_spearman"],
            "std_val_spearman": kfold["std_val_spearman"],
        },
        "unseen_drug_result": {
            "note": "Held-out drugs never seen during training - hardest, most meaningful test",
            "test_spearman": unseen["test_spearman"],
            "test_mse": unseen["test_mse"],
            "n_test_drugs": unseen["n_test_drugs"],
        },
    }

    with open("results/tables/arm_a_final_submission.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nSaved to results/tables/arm_a_final_submission.json")

if __name__ == "__main__":
    build_submission_summary()