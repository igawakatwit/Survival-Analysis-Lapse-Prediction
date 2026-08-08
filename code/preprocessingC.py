import pandas as pd

def preprocess_data(df_raw : pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    df = df.drop_duplicates()

    cat_var = [
        "gender", "type_policy", "type_policy_dg", "type_product", "reimbursement",
        "new_business", "distribution_channel", "lapse", "period", "C_H", "C_GI",
        "C_II", "C_IE_P", "C_IE_S", "C_IE_T", "C_GE_P", "C_GE_S", "C_GE_T", "C_C"
    ]
    df[cat_var] = df[cat_var].astype("category")

    df["censor_date"] = pd.to_datetime(df["period"].astype(str))
    df["end_date"] = df["date_lapse_insured"].fillna(df["censor_date"])
    df["duration_days"] = (df["end_date"] - df["date_effect_insured"]).dt.days
    df["duration_years"] = (df["duration_days"]) / 365.25
    df["duration_years"] = df["duration_years"].clip(lower=0)

    df["event"] = df["lapse"].apply(lambda x: 1 if x == 1 else 0)

    df = df.drop(columns = ["ID", "ID_policy", "ID_insured"])
    df = df.drop(columns = ["IICIMUN", "IICIPROV"])

    socio_cols = ["C_H", "C_GI", "C_II", "C_IE_P", "C_IE_S", "C_IE_T",
                  "C_GE_P", "C_GE_S", "C_GE_T", "C_C"
    ]
    df = df.dropna(subset = socio_cols)
    return df