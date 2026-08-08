import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_single_seed_subset(df, cph_model_fn, rsf_model_fn, coxtime_model_fn, time_col, feature_cols, event_col, frac, seed):
    df_small = df.sample(frac=frac, random_state=seed)

    train_df, test_df = train_test_split(
        df_small, test_size=0.2, stratify=df_small[event_col], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.25, stratify=train_df[event_col], random_state=seed
    )

    continuous_cols = [
        "seniority_insured", "seniority_policy", "reimbursement",
        "cost_claims_year", "premium_log_centered",
        "n_insured_pc", "n_insured_mun", "n_insured_prov"
    ]

    scaler = StandardScaler()
    train_df[continuous_cols] = scaler.fit_transform(train_df[continuous_cols])
    val_df[continuous_cols] = scaler.transform(val_df[continuous_cols])
    test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])

    results = {}

    #cph
    cph, _ = cph_model_fn(train_df, val_df, test_df)
    results["coxph"] = cph.score(test_df)

    #rsf
    rsf, X_val, y_val, X_test, y_test, _ = rsf_model_fn(train_df, val_df, test_df, feature_cols)
    results["rsf"] = rsf.score(X_test, y_test)

    #coxtime
    model_ct, scaler_ct, c_index_ct = coxtime_model_fn(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols
    )
    results["coxtime"] = c_index_ct

    return results


def run_multi_seed_robustness(df, cph_model_fn, rsf_model_fn, coxtime_model_fn, feature_cols,time_col="duration_years", event_col="event", frac=0.25, seeds=[42, 21, 84]):
    all_results = {}

    for seed in seeds:
        seed_results = run_single_seed_subset(
            df=df,
            cph_model_fn=cph_model_fn,
            rsf_model_fn=rsf_model_fn,
            coxtime_model_fn=coxtime_model_fn,
            feature_cols=feature_cols,
            time_col=time_col,
            event_col=event_col,
            frac=frac,
            seed=seed
        )
        all_results[seed] = seed_results

    return all_results
