import sqlite3
import pandas as pd
import os

CHEMBL_DB = "data/raw/chembl/chembl_37/chembl_37_sqlite/chembl_37.db"


def get_chembl_name_to_smiles():
    conn = sqlite3.connect(CHEMBL_DB)
    query = """
    SELECT
        md.chembl_id,
        md.pref_name,
        cs.canonical_smiles
    FROM molecule_dictionary md
    JOIN compound_structures cs ON md.molregno = cs.molregno
    WHERE md.pref_name IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_chembl_synonyms():
    conn = sqlite3.connect(CHEMBL_DB)
    query = """
    SELECT
        md.chembl_id,
        ms.synonyms,
        cs.canonical_smiles
    FROM molecule_synonyms ms
    JOIN molecule_dictionary md ON ms.molregno = md.molregno
    JOIN compound_structures cs ON ms.molregno = cs.molregno
    WHERE ms.synonyms IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_gdsc2_drug_names(gdsc2_path="data/raw/gdsc2/GDSC2_fitted_dose_response_27Oct23.xlsx"):
    df = pd.read_excel(gdsc2_path)
    return df[["DRUG_ID", "DRUG_NAME"]].drop_duplicates()


def match_drugs_to_chembl(gdsc2_drugs: pd.DataFrame, chembl_df: pd.DataFrame):
    chembl_df = chembl_df.copy()
    chembl_df["name_norm"] = chembl_df["pref_name"].str.upper().str.strip()

    gdsc2_drugs = gdsc2_drugs.copy()
    gdsc2_drugs["name_norm"] = gdsc2_drugs["DRUG_NAME"].str.upper().str.strip()

    merged = gdsc2_drugs.merge(chembl_df, on="name_norm", how="left")
    matched = merged[merged["canonical_smiles"].notna()]
    unmatched = merged[merged["canonical_smiles"].isna()]

    print(f"Matched: {len(matched)} / {len(gdsc2_drugs)} GDSC2 drugs")
    print(f"Unmatched: {len(unmatched)}")
    if len(unmatched) > 0:
        print("\nSample unmatched drug names:")
        print(unmatched["DRUG_NAME"].head(15).tolist())

    return matched[["DRUG_ID", "DRUG_NAME", "chembl_id", "canonical_smiles"]], unmatched[["DRUG_ID", "DRUG_NAME"]]


def match_via_synonyms(unmatched_drugs: pd.DataFrame, synonyms_df: pd.DataFrame):
    synonyms_df = synonyms_df.copy()
    synonyms_df["name_norm"] = synonyms_df["synonyms"].str.upper().str.strip()

    unmatched_drugs = unmatched_drugs.copy()
    unmatched_drugs["name_norm"] = unmatched_drugs["DRUG_NAME"].str.upper().str.strip()

    merged = unmatched_drugs.merge(synonyms_df, on="name_norm", how="left")
    merged = merged.drop_duplicates(subset=["DRUG_ID"], keep="first")

    matched = merged[merged["canonical_smiles"].notna()]
    still_unmatched = merged[merged["canonical_smiles"].isna()]

    print(f"Matched via synonyms: {len(matched)} / {len(unmatched_drugs)}")
    print(f"Still unmatched: {len(still_unmatched)}")
    if len(still_unmatched) > 0:
        print("\nSample still-unmatched names:")
        print(still_unmatched["DRUG_NAME"].head(20).tolist())

    return matched[["DRUG_ID", "DRUG_NAME", "chembl_id", "canonical_smiles"]], still_unmatched[["DRUG_ID", "DRUG_NAME"]]


def fuzzy_match_remaining(still_unmatched: pd.DataFrame, chembl_df: pd.DataFrame, threshold: int = 90):
    from rapidfuzz import process, fuzz

    choices = chembl_df["pref_name"].dropna().unique().tolist()
    results = []
    for _, row in still_unmatched.iterrows():
        match = process.extractOne(row["DRUG_NAME"], choices, scorer=fuzz.WRatio)
        if match and match[1] >= threshold:
            matched_row = chembl_df[chembl_df["pref_name"] == match[0]].iloc[0]
            results.append({
                "DRUG_ID": row["DRUG_ID"],
                "DRUG_NAME": row["DRUG_NAME"],
                "chembl_id": matched_row["chembl_id"],
                "canonical_smiles": matched_row["canonical_smiles"],
                "matched_to": match[0],
                "match_score": match[1],
            })
    matched_df = pd.DataFrame(results)
    print(f"Fuzzy-matched: {len(matched_df)} / {len(still_unmatched)}")
    if len(matched_df) > 0:
        print(matched_df[["DRUG_NAME", "matched_to", "match_score"]])
    return matched_df


if __name__ == "__main__":
    chembl_df = get_chembl_name_to_smiles()
    print("ChEMBL compounds loaded:", chembl_df.shape)

    gdsc2_drugs = load_gdsc2_drug_names()
    print("GDSC2 drugs:", gdsc2_drugs.shape)

    matched, unmatched = match_drugs_to_chembl(gdsc2_drugs, chembl_df)

    synonyms_df = get_chembl_synonyms()
    print("\nChEMBL synonyms loaded:", synonyms_df.shape)

    matched2, still_unmatched = match_via_synonyms(unmatched, synonyms_df)

    matched_all = pd.concat([matched, matched2], ignore_index=True)
    print(f"\nTotal matched (name + synonym): {len(matched_all)} / {len(gdsc2_drugs)}")

    fuzzy_matched = fuzzy_match_remaining(still_unmatched, chembl_df)
    if len(fuzzy_matched) > 0:
        matched_all = pd.concat(
            [matched_all, fuzzy_matched[["DRUG_ID", "DRUG_NAME", "chembl_id", "canonical_smiles"]]],
            ignore_index=True
        )
        still_unmatched = still_unmatched[~still_unmatched["DRUG_ID"].isin(fuzzy_matched["DRUG_ID"])]
        print(f"\nTotal matched after fuzzy pass: {len(matched_all)} / {len(gdsc2_drugs)}")

    # remove known-bad fuzzy matches (verified manually — algorithm matched on
    # meaningless short substrings like "17" or "13" rather than real compound identity)
    bad_fuzzy_matches = ["JAK_8517", "123138", "BEN", "GNE-317"]
    bad_rows = matched_all[matched_all["DRUG_NAME"].isin(bad_fuzzy_matches)]
    matched_all = matched_all[~matched_all["DRUG_NAME"].isin(bad_fuzzy_matches)]
    still_unmatched = pd.concat([still_unmatched, bad_rows[["DRUG_ID", "DRUG_NAME"]]], ignore_index=True)
    print(f"\nRemoved {len(bad_rows)} bad fuzzy matches, moved back to unmatched")
    print(f"Final matched count: {len(matched_all)} / {len(gdsc2_drugs)}")

    os.makedirs("data/processed", exist_ok=True)
    matched_all.to_csv("data/processed/drug_id_map.csv", index=False)
    still_unmatched.to_csv("data/processed/drug_id_map_unmatched.csv", index=False)
    print("\nSaved matched drugs to data/processed/drug_id_map.csv")
    print("Saved still-unmatched drugs to data/processed/drug_id_map_unmatched.csv")