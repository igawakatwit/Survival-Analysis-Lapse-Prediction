import pandas as pd
import numpy as np

def feature_engineering(df):
    #calls all functions
    df = age_group(df)
    df = premium_features(df)
    df = ses_tiers(df)
    df = utilization_features(df)
    df = policy_structure_features(df)
    df = distribution_channel_features(df)
    df = drop_reference_dummies(df)


    #dropping raw columns that won't be used in the model
    cols_to_drop = [
        "date_effect_insured", "date_lapse_insured",
        "date_effect_policy", "date_lapse_policy",
        "censor_date", "end_date",
        "year_effect_insured", "year_lapse_insured",
        "year_effect_policy", "year_lapse_policy",
        "duration_days", "lapse",
        "type_policy", "type_policy_dg",
        "age_group", "premium_decile", "util_bucket",
        "C_GI", "C_II", "C_IE_P", "C_IE_S", "C_IE_T",
        "C_GE_P", "C_GE_S", "C_GE_T",
        "premium", "period", "type_product", "C_H", "C_C",
        "n_medical_services", "age", "distribution_channel",
        "exposure_time", "C_GI_tier", "C_II_tier", "C_IE_P_tier",
        "C_IE_S_tier", "C_IE_T_tier", "C_GE_P_tier", "C_GE_S_tier",
        "C_GE_T_tier",  "seniority_insured", "seniority_policy"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    #convert categorical to numeric
    df["gender"] = df["gender"].map({"M": 1, "F": 0}).astype(int)
    df["reimbursement"] = df["reimbursement"].map({"Yes": 1, "No": 0}).astype(int)
    df["new_business"] = df["new_business"].map({"Yes": 1, "No": 0}).astype(int)

    #convert boolean one-hot columns to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df

#creates behavior based age bins
def age_group(df, age_col = "age"):
    bins = [-np.inf, 18, 25, 34, 45, 55, 65, np.inf]
    labels = [
        "age_0_18", "age_19_25", "age_26_34",
        "age_36_45", "age_46_55", "age_56_65",
        "age_65_plus"
    ]
    df["age_group"] = pd.cut(df[age_col], bins=bins, labels=labels)
    #one-hot-encoding the new categorical variables
    age_dummies = pd.get_dummies(df["age_group"], prefix="", prefix_sep="")
    df = pd.concat([df, age_dummies], axis=1)
    return df

def premium_features(df, premium_col="premium"):
    df["premium_adjusted"] = df[premium_col].clip(lower=1)
    #applies log transformation which reduces skew and compresses large premiums
    df["premium_log"] = np.log(df["premium_adjusted"])
    #creates premium deciles (10 equally sized groups)
    df["premium_decile"] = pd.qcut(
        df[premium_col],
        q=10,
        labels=[f"prem_d{i}" for i in range(1, 11)],
        duplicates="drop"
    )
    #one hot encodes the premium deciles
    prem_dummies = pd.get_dummies(df["premium_decile"], prefix="", prefix_sep="")
    df = pd.concat([df, prem_dummies], axis=1)
    #centers the log premium, how far away from average premium you are
    df["premium_log_centered"] = df["premium_log"] - df["premium_log"].mean()

    if "premium_adjusted" in df.columns:
        df = df.drop(columns=["premium_adjusted"])
    return df


def ses_tiers(df):
    ses_cols = ["C_GI","C_II", "C_IE_P", "C_IE_S", "C_IE_T",
                "C_GE_P", "C_GE_S", "C_GE_T"]
    for col in ses_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
        #makes sure each column has at least 5 unique values
        if df[col].nunique() < 5:
            print(f"Skipping {col} - not enough unique numeric values for 5 tiers")
            continue
        try:
            #splits each SES variable into five equal-sized groups
            df[f"{col}_tier"] = pd.qcut(
                df[col],
                q=5,
                labels = [f"{col}_T1", f"{col}_T2", f"{col}_T3", f"{col}_T4", f"{col}_T5"],
                duplicates = "drop"
            )
            #one hot-encoding new tier variables
            ses_dummies = pd.get_dummies(df[f"{col}_tier"], prefix="", prefix_sep="")
            df = pd.concat([df, ses_dummies], axis=1)
        except Exception as e:
            print(f"Skipping {col} due to qcut error: {e}")
    return df

def utilization_features(df, util_col = "n_medical_services"):
    df[util_col] = df[util_col].fillna(0)
    #policies that used 0 medical services are put into their own bucket
    df["util_bucket"] = "U0"
    nonzero = df[df[util_col] > 0][util_col]

    if nonzero.nunique() >= 4:
        # splits non-zero utilization into quartiles
        df.loc[df[util_col] > 0, "util_bucket"] = pd.qcut(
            nonzero,
            q=4,
            labels=["U1", "U2", "U3", "U4"],
            duplicates="drop"
        )
    else:
        print("Not enough unique non-zero utilization values to create quantile bins.")

    #one hot encodes utilization buckets
    util_dummies = pd.get_dummies(df["util_bucket"], prefix="", prefix_sep="")
    df = pd.concat([df, util_dummies], axis=1)
    return df

def policy_structure_features(df, policy_col="type_policy", policy_dg_col="type_policy_dg"):
    #one hot encodes the policy distribution groups, creates dummy variables for I, S, C1, C2, C3, C4
    policy_dg_dummies = pd.get_dummies(df[policy_dg_col], prefix="polDG")
    df = pd.concat([df, policy_dg_dummies], axis=1)

    #creates interaction terms with centered premium
    #lets the model learn things like 'high premium SE policies behave differently than low premium SE policies'
    if "premium_log_centered" in df.columns:
        for col in policy_dg_dummies.columns:
            df[f"{col}_x_premium_log_centered"] = df[col] * df["premium_log_centered"]

    if "premium_log" in df.columns:
        df = df.drop(columns=["premium_log"])

    return df


def distribution_channel_features(df):
    df["distribution_channel"] = df["distribution_channel"].astype(str)
    #one-hot encodes distribution channel (agent, commercial, etc)
    dummies = pd.get_dummies(df["distribution_channel"], prefix = "dist")
    df = pd.concat([df, dummies], axis=1)
    return df

def drop_reference_dummies(df):
    #removes one dummy variable from every one hot encoded group
    #dropping one column allows for the cox model to have a valid baseline category and to avoid multicollinearity
    reference_cols = [
        "age_0_18",
        "prem_d1",
        "C_GI_T1", "C_II_T1", "C_IE_P_T1", "C_IE_S_T1",
        "C_IE_T_T1", "C_GE_P_T1", "C_GE_S_T1", "C_GE_T_T1",
        "U0",
        "dist_A", "polDG_I_x_premium_log_centered",
        "polDG_I"
    ]

    df = df.drop(columns=[c for c in reference_cols if c in df.columns])
    return df
