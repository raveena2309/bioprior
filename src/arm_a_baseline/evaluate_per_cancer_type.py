import pandas as pd
from scipy.stats import spearmanr

def evaluate_per_cancer_type():
    val_preds = pd.read_csv("results/tables/arm_a_val_predictions.csv")
    cellline_map = pd.read_csv("data/processed/cellline_id_map.csv")
    gdsc2 = pd.read_excel("data/raw/gdsc2/GDSC2_fitted_dose_response_27Oct23.xlsx")

    candidate_cols = [c for c in gdsc2.columns if "TCGA" in c.upper() or "CANCER" in c.upper() or "TISSUE" in c.upper()]
    print("Candidate cancer-type columns found:", candidate_cols)
    if not candidate_cols:
        print("None found automatically — full GDSC2 column list:")
        print(list(gdsc2.columns))
        return
    cancer_col = candidate_cols[0]
    print(f"Using column: {cancer_col}")

    gdsc2_lookup = gdsc2[["SANGER_MODEL_ID", cancer_col]].drop_duplicates(subset=["SANGER_MODEL_ID"])

    merged = val_preds.merge(cellline_map, left_on="cell_line_id", right_on="ccle_model_id", how="left")
    merged = merged.merge(gdsc2_lookup, left_on="sanger_model_id", right_on="SANGER_MODEL_ID", how="left")
    print(f"Rows with no cancer-type match: {merged[cancer_col].isna().sum()} / {len(merged)}")

    results = []
    for cancer_type, group in merged.dropna(subset=[cancer_col]).groupby(cancer_col):
        if len(group) < 30:  # skip groups too small to be meaningful
            continue
        corr = spearmanr(group["y_true"], group["y_pred"]).correlation
        results.append({"cancer_type": cancer_type, "n_pairs": len(group), "spearman": corr})

    results_df = pd.DataFrame(results).sort_values("spearman", ascending=False)
    results_df.to_csv("results/tables/arm_a_per_cancer_type.csv", index=False)
    print(results_df)
    return results_df

if __name__ == "__main__":
    evaluate_per_cancer_type()