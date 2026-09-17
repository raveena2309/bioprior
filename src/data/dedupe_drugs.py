import pandas as pd

def dedupe_drug_map(
    drug_map_path="data/processed/drug_id_map.csv",
    out_path="data/processed/drug_id_map_deduped.csv",
):
    df = pd.read_csv(drug_map_path)
    print(f"Before dedup: {len(df)} drug entries, {df['DRUG_NAME'].nunique()} unique names")

    # keep the first DRUG_ID seen for each drug name, build a mapping from
    # every duplicate ID to that canonical ID
    df_sorted = df.sort_values("DRUG_ID")
    canonical = df_sorted.drop_duplicates(subset="DRUG_NAME", keep="first")

    id_to_canonical = {}
    for name in df["DRUG_NAME"].unique():
        ids_for_name = df[df["DRUG_NAME"] == name]["DRUG_ID"].tolist()
        canonical_id = min(ids_for_name)
        for did in ids_for_name:
            id_to_canonical[did] = canonical_id

    print(f"After dedup: {len(canonical)} unique drugs")
    duplicates_found = {k: v for k, v in id_to_canonical.items() if k != v}
    print(f"Duplicate ID -> canonical ID mappings: {len(duplicates_found)}")
    for dup_id, canon_id in duplicates_found.items():
        print(f"  {dup_id} -> {canon_id}")

    canonical.to_csv(out_path, index=False)
    print(f"\nSaved deduped drug map to {out_path}")

    # also save the ID-to-canonical mapping itself, so training_table.csv can be rebuilt using it
    mapping_df = pd.DataFrame(list(id_to_canonical.items()), columns=["DRUG_ID", "canonical_drug_id"])
    mapping_df.to_csv("data/processed/drug_id_canonical_map.csv", index=False)
    print("Saved ID canonicalization map to data/processed/drug_id_canonical_map.csv")

    return canonical, id_to_canonical

if __name__ == "__main__":
    dedupe_drug_map()