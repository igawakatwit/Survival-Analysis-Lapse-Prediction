import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from edaC import exploratory_analysis
from preprocessingC import preprocess_data
from featureengineeringC import feature_engineering
from baselineC import check_ph_assumptions, cox_model, compute_brier_score, compute_integrated_brier_score
from ML_modelsC import random_survival_forest
from visualsC import predict_coxph_survival_at_time,shap_coxph, predict_coxtime_survival_at_time, predict_rsf_survival_at_time, shap_rsf,shap_coxtime, calibration_plot, plot_coxtime_fan_chart, KM_curves, lapse_timing_distribution, exploratory_graphs, survival_curves_baseline, compare_policy_survival, plot_rsf_fan_chart, plot_cox_fan_chart
from neural_modelC import run_coxtime
from robustness_testC import run_multi_seed_robustness


import numpy as np
if __name__ == "__main__":
    filepath = "C:/Users/15623/Desktop/Health Insurance Dataset.xlsx"
    df_raw = pd.read_excel(filepath)

    #exploratory analysis on raw data
    exploratory_analysis(df_raw)

    #preprocessing
    df = preprocess_data(df_raw)

    #exploratory analysis on cleaned data
    exploratory_graphs(df)
    lapse_timing_distribution(df)
    KM_curves(df)

    #feature engineering
    df = feature_engineering(df)
    print(df.columns)

    #splits into train, test, validation, uses stratification to make sure each split has the same event ratio
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["event"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.25, stratify=train_df["event"], random_state=42
    )

    #checks VIF for each feature to prevent perfect multicollinearity
    X = df.drop(columns=["duration_years", "event"])
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif.sort_values("VIF", ascending=False))

    #scale continuous variables, cox uses gradient based optimization so need variables on same scale
    continuous_cols = [
        "reimbursement",
        "cost_claims_year",
        "premium_log_centered",
        "n_insured_pc",
        "n_insured_mun",
        "n_insured_prov"
    ]

    scaler = StandardScaler()
    train_df[continuous_cols] = scaler.fit_transform(train_df[continuous_cols])
    val_df[continuous_cols] = scaler.transform(val_df[continuous_cols])
    test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])

    #fit the COX model
    cph, features = cox_model(train_df, val_df, test_df)

    times, brier = compute_brier_score(train_df, test_df, cph, features)
    print("Mean Brier Score:", np.mean(brier))

    ibs = compute_integrated_brier_score(train_df, test_df, cph, features)
    print("Integrated Brier Score:", ibs)

    #check assumptions
    check_ph_assumptions(
        train_df,
        cph,
        show_plots=True,
        columns=[
            "seniority_insured",
            "seniority_policy",
            "premium_log_centered",
            "cost_claims_year"
        ]
    )

    survival_curves_baseline(test_df, cph)

    #Fit the RSF Model
    majority_features = ["cost_claims_year"]
    print("\nRunning RSF with majority features only...")

    rsf_majority, X_val_mj, y_val_mj, X_test_mj, y_test_mj, rsf_time_grid_mj = random_survival_forest(
        train_df,
        val_df,
        test_df,
        majority_features
    )

    rsf_majority_cindex = rsf_majority.score(X_test_mj, y_test_mj)
    print(f"RSF Test C-index (majority features only): {rsf_majority_cindex:.4f}")

    rsf, X_val, y_val, X_test, y_test, rsf_time_grid = random_survival_forest(
        train_df, val_df, test_df, features
    )

    feature_cols = [
        col for col in df.columns
        if col not in ["duration_years", "event"]
    ]

    model_ct, scaler_ct, c_train, c_val, c_test = run_coxtime(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols
    )

    X_test_scaled = scaler_ct.transform(
        test_df[feature_cols].values.astype("float32")
    )

    plot_rsf_fan_chart(rsf, test_df)
    plot_cox_fan_chart(cph, test_df)
    plot_coxtime_fan_chart(model_ct, X_test_scaled, n_samples=250)

    t_star = 10.0
    if t_star >= test_df["duration_years"].max():
        raise ValueError(
            f"time horizon {t_star} must be below the maximum "
            f"test follow-up time of {test_df['duration_years'].max():.4f}"
        )

    #Use the same test observations for all three models
    calibration_size = min(5000, len(test_df))
    calibration_df = test_df.sample(
        n=calibration_size,
        random_state=42
    ).copy()

    X_coxph = calibration_df[features].copy()
    X_rsf = calibration_df[features].copy()
    X_coxtime_input = calibration_df[feature_cols].copy()

    X_coxtime_scaled = scaler_ct.transform(
        X_coxtime_input.to_numpy(dtype="float32")
    ).astype("float32")

    coxph_surv_at_t = predict_coxph_survival_at_time(cph, X_coxph, t_star)
    rsf_surv_at_t = predict_rsf_survival_at_time(rsf, X_rsf, t_star)
    coxtime_surv_at_t = predict_coxtime_survival_at_time(
        model_ct,
        X_coxtime_scaled,
        t_star,
        batch_size=500
    )

    print("CoxPH predictions shape:", coxph_surv_at_t.shape)
    print("RSF predictions shape:", rsf_surv_at_t.shape)
    print("Cox-Time predictions shape:", coxtime_surv_at_t.shape)

    model_predictions = {
        "CoxPH": coxph_surv_at_t,
        "RSF": rsf_surv_at_t,
        "Cox-Time": coxtime_surv_at_t
    }

    calibration_plot(
        df=calibration_df,
        model_predictions=model_predictions,
        model_names=["CoxPH", "RSF", "Cox-Time"],
        time_horizon=t_star,
        n_bins=10
    )

    shap_coxph(cph_model=cph, X=X_coxph, time_horizon=t_star)
    shap_rsf(rsf_model=rsf, X=X_rsf, time_horizon=t_star)
    shap_coxtime(
        coxtime_model=model_ct,
        scaler=scaler_ct,
        X=X_coxtime_input,
        time_horizon=t_star
    )

    multi_seed_results = run_multi_seed_robustness(
        df=df,
        cph_model_fn=cox_model,
        rsf_model_fn=random_survival_forest,
        coxtime_model_fn=run_coxtime,
        feature_cols=feature_cols,
        time_col="duration_years",
        event_col="event",
        frac=0.25,
        seeds=[42, 21, 84]
    )

    print("\nMulti-Seed Robustness Check (25% subset):")
    for seed, result_dict in multi_seed_results.items():
        print(f"\nSeed {seed}:")
        for model_name, cidx in result_dict.items():
            print(f"  {model_name}: {cidx:.4f}")

    compare_policy_survival(
        cph=cph,
        rsf=rsf,
        coxtime_model=model_ct,
        test_df=test_df,
        X_test=X_test_scaled,
        feature_cols=feature_cols,
        policy_index=150
    )

    compare_policy_survival(
        cph=cph,
        rsf=rsf,
        coxtime_model=model_ct,
        test_df=test_df,
        X_test=X_test_scaled,
        feature_cols=feature_cols,
        policy_index=200
    )



