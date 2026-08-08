import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage.color.rgb_colors import darkgreen, lightblue
from lifelines import NelsonAalenFitter
import pandas as pd
from lifelines import KaplanMeierFitter
import numpy as np
import shap

def shap_coxph(cph, X, time_horizon):
    feature_names = list(X.columns)
    def predict_risk(X_array):
        X_frame = pd.DataFrame(X_array, columns=feature_names)
        survival = predict_coxph_survival_at_time(cph, X_frame, time_horizon)
        return 1.0 - survival

    kernel_shap_summary(
        predict_function=predict_risk,
        X=X,
        model_name=f"CoxPH {time_horizon:.0f}-Year Risk",
        background_size=20,
        explanation_size=75,
        nsamples=100
    )

def shap_rsf(rsf_model, X, time_horizon):
    feature_names = list(X.columns)
    def predict_risk(X_array):
        X_frame = pd.DataFrame(X_array, columns=feature_names)
        survival = predict_rsf_survival_at_time(rsf_model, X_frame, time_horizon)
        return 1.0 - survival

    kernel_shap_summary(
        predict_function=predict_risk,
        X=X,
        model_name=f"RSF {time_horizon:.0f}-Year Risk",
        background_size=20,
        explanation_size=75,
        nsamples=100
    )

def shap_coxtime(cph, scaler, X, time_horizon):
    feature_names = list(X.columns)
    def predict_risk(X_array):
        X_frame = pd.DataFrame(X_array, columns=feature_names)
        X_scaled = scaler.transform(X_frame.to_numpy(dtype="float32")).astype("float32")
        survival = predict_coxtime_survival_at_time(cph, X_scaled, time_horizon, batch_size=500)
        return 1.0 - survival

    kernel_shap_summary(
        predict_function=predict_risk,
        X=X,
        model_name=f"Cox-Time {time_horizon:.0f}-Year Risk",
        background_size=20,
        explanation_size=75,
        nsamples=100
    )

def exploratory_graphs(df):
    #plot showing distribution of policy duration (years)
    plt.figure(figsize=(10, 6))
    sns.histplot(df["duration_years"], bins=50, kde=True, color = lightblue)

    plt.title("Distribution of Policy Durations (Years)")
    plt.xlabel("Duration (years)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    #plot showing distribution of underwriting age
    plt.figure(figsize=(10, 6))
    sns.histplot(df["age"], bins=50, kde=True, color = darkgreen)

    plt.title("Distribution of Age")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    #scatterplot showing age vs duration colored by event
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="age",
        y="duration_years",
        hue="event",
        palette={0: "gray", 1: "red"},
        alpha=0.5
    )
    plt.title("Age vs Duration (Colored by Lapse Event)")
    plt.xlabel("Age")
    plt.ylabel("Duration (years)")
    plt.tight_layout()

    #plot showing age distribution by lapse event
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="event", y="age")
    plt.title("Age Distribution by Lapse Event")
    plt.xlabel("Event (1 = Lapse, 0 = Censored)")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.show()

def lapse_timing_distribution(df):
    #filter only lapse events
    df_events = df[df["event"] == 1]

    plt.figure(figsize=(10, 6))
    sns.histplot(df_events["duration_years"], bins=40, kde=True, color="red")
    plt.title("Distribution of Lapse Timing (Events Only)")
    plt.xlabel("Duration (years)")
    plt.ylabel("Count")
    plt.show()

    #Kaplan-Meier Survival curve, shows overall survival probability over time
    km = KaplanMeierFitter()

    plt.figure(figsize=(10, 6))
    km.fit(durations=df["duration_years"], event_observed=df["event"])
    km.plot(ci_show=True)

    plt.title("Kaplan–Meier Survival Curve")
    plt.xlabel("Duration (years)")
    plt.ylabel("Survival Probability")
    plt.show()

    #hazard function, shows how hazard accumulates
    naf = NelsonAalenFitter()

    plt.figure(figsize=(10, 6))
    naf.fit(df["duration_years"], event_observed=df["event"])
    naf.plot()

    plt.title("Nelson–Aalen Cumulative Hazard Function")
    plt.xlabel("Duration (years)")
    plt.ylabel("Cumulative Hazard")
    plt.show()

def KM_curves(df):
    plt.figure(figsize=(12, 8))
    km = KaplanMeierFitter()

    for t in df["type_policy"].cat.categories:
        subset = df[df["type_policy"] == t]
        km.fit(subset["duration_years"], subset["event"], label=str(t))
        km.plot(ci_show=False)

    plt.title("KM Survival Curves by Policy Type")
    plt.xlabel("Duration (years)")
    plt.ylabel("Survival Probability")
    plt.legend(title="Policy Type")
    plt.show()

    df["premium_quartile"] = pd.qcut(df["premium"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    plt.figure(figsize=(12, 8))
    km = KaplanMeierFitter()

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        subset = df[df["premium_quartile"] == q]
        km.fit(subset["duration_years"], subset["event"], label=q)
        km.plot(ci_show=False)

    plt.title("KM Survival Curves by Premium Quartile")
    plt.xlabel("Duration (years)")
    plt.ylabel("Survival Probability")
    plt.legend(title="Premium Quartile")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="event", palette=["gray", "red"])
    plt.title("Count of Censored vs Lapsed Policies")
    plt.xlabel("Event (0 = Censored, 1 = Lapse)")
    plt.ylabel("Count")
    plt.xticks([0, 1], ["Censored", "Lapse"])
    plt.show()

def survival_curves_baseline(test_df, cph):
    #plot a survival curve for a single example policy
    example = test_df.sample(1, random_state=42)
    surv = cph.predict_survival_function(example)
    plt.figure(figsize=(8, 5))
    plt.plot(surv.index, surv.values)
    plt.title("Predicted Survival Curve for Example Policy")
    plt.xlabel("Years")
    plt.ylabel("Survival Probability (1 - Lapse)")
    plt.grid(True)
    plt.show()

    #sample 200 policies from each group to get stable averages
    group1 = test_df[test_df["U4"] == 1].sample(200)
    group2 = test_df[test_df["U1"] == 1].sample(200)
    #compute group level survival curves
    surv1 = cph.predict_survival_function(group1).mean(axis=1)
    surv2 = cph.predict_survival_function(group2).mean(axis=1)
    #plot comparison
    plt.figure(figsize=(8, 5))
    plt.plot(surv1.index, surv1, label="High Utilization (U4)")
    plt.plot(surv2.index, surv2, label="Low Utilization (U1)")
    plt.title("Group Survival Curves")
    plt.xlabel("Years")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rsf_fan_chart(rsf, df, n_samples=200):
    sample_df = df.sample(n_samples, random_state=42)
    plt.figure(figsize=(12, 8))

    for _, row in sample_df.iterrows():
        x_row = row.drop(["duration_years", "event"]).to_frame().T
        surv_list = rsf.predict_survival_function(x_row)

        fn = surv_list[0]
        times = fn.x
        values = fn(times)

        plt.plot(times, values, color="steelblue", alpha=0.1)

    plt.title("RSF Survival Curve Fan Chart (Model Overview)")
    plt.xlabel("Time (years)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.show()


def plot_cox_fan_chart(cph, df, n_samples=200):
    sample_df = df.sample(n_samples, random_state=42)
    plt.figure(figsize=(12, 8))

    for _, row in sample_df.iterrows():
        x_row = row.to_frame().T
        surv = cph.predict_survival_function(x_row)

        times = surv.index.values
        values = surv.iloc[:, 0].values

        plt.plot(times, values, color="darkred", alpha=0.1)

    plt.title("Cox PH Survival Curve Fan Chart (Model Overview)")
    plt.xlabel("Time (years)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.show()

def plot_coxtime_fan_chart(model, X_test, n_samples=200):
    idx = np.random.choice(len(X_test), size=n_samples, replace=False)

    plt.figure(figsize=(12, 8))

    for i in idx:
        surv_df = model.predict_surv_df(X_test[i:i+1])
        times = surv_df.index.values
        values = surv_df.iloc[:, 0].values

        plt.plot(times, values, color="darkorange", alpha=0.08)

    plt.title("Cox-Time Survival Curve Fan Chart (Model Overview)")
    plt.xlabel("Time (Years)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.show()

def compare_policy_survival(cph, rsf, coxtime_model, test_df, X_test, feature_cols, policy_index=0):
    #cph
    cph_surv = cph.predict_survival_function(test_df.iloc[[policy_index]])
    cph_times = cph_surv.index.values
    cph_probs = cph_surv.iloc[:, 0].values

    #rsf
    rsf_input = test_df[feature_cols].iloc[[policy_index]]
    rsf_surv_fn = rsf.predict_survival_function(rsf_input)[0]

    rsf_times = rsf_surv_fn.x
    rsf_probs = rsf_surv_fn(rsf_times)

    #cox-time
    ct_surv_df = coxtime_model.predict_surv_df(X_test[policy_index:policy_index+1])
    ct_times = ct_surv_df.index.values
    ct_probs = ct_surv_df.iloc[:, 0].values

    plt.figure(figsize=(8, 5))

    plt.plot(cph_times, cph_probs, linewidth=2, label="Cox PH")
    plt.plot(rsf_times, rsf_probs, linewidth=2, label="RSF")
    plt.plot(ct_times, ct_probs, linewidth=2, label="Cox-Time")

    plt.title(f"Policy {policy_index}: Survival Curve Comparison")
    plt.xlabel("Time (Years)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def calibration_plot(df, model_predictions, model_names, time_horizon, n_bins=10):
    plt.figure(figsize=(8, 6))
    for model_name in model_names:
        predictions = np.asarray(model_predictions[model_name], dtype=float)

        if len(predictions) != len(df):
            raise ValueError(
                f"{model_name} has {len(predictions)} predictions, "
                f"but the calibration data has {len(df)} rows."
            )

        work_df = df[["duration_years", "event"]].copy()
        work_df["pred"] = np.clip(predictions,0.0,1.0)

        unique_predictions = work_df["pred"].nunique()

        if unique_predictions < 2:
            print(
                f"Skipping {model_name}: "
                "predictions contain fewer than two unique values."
            )
            continue

        bins_to_use = min(n_bins, unique_predictions)
        work_df["bin"] = pd.qcut(work_df["pred"], q=bins_to_use, duplicates="drop")
        observed_survival = []
        predicted_survival = []

        for _, group in work_df.groupby("bin", observed=True):
            if len(group) == 0:
                continue
            km = KaplanMeierFitter()
            km.fit(durations=group["duration_years"], event_observed=group["event"])

            observed = (km.survival_function_at_times([time_horizon]).iloc[0])
            predicted = group["pred"].mean()
            observed_survival.append(observed)
            predicted_survival.append(predicted)

        calibration_results = pd.DataFrame({
            "predicted": predicted_survival,
            "observed": observed_survival
        }).sort_values("predicted")

        plt.plot(
            calibration_results["predicted"],
            calibration_results["observed"],
            marker="o",
            label=model_name
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    plt.xlabel("Mean Predicted Survival Probability")
    plt.ylabel("Kaplan–Meier Observed Survival Probability")
    plt.title(
        f"Calibration at {time_horizon:.1f} Years"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def kernel_shap_summary(predict_function, X, model_name, background_size=20, explanation_size=75, nsamples=100):
    X = X.copy()
    background_size = min(background_size, len(X))
    explanation_size = min(explanation_size, len(X))

    background = X.sample(n=background_size, random_state=42)
    explanation_data = X.sample(n=explanation_size, random_state=123)

    explainer = shap.KernelExplainer(predict_function, background.to_numpy())

    shap_values = explainer.shap_values(explanation_data.to_numpy(), nsamples=nsamples)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, explanation_data, show=False)

    plt.title(f"{model_name} SHAP Summary")
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, explanation_data, plot_type="bar", show=False)
    plt.title(f"{model_name} Mean Absolute SHAP Values")

    plt.tight_layout()
    plt.show()

def predict_coxtime_survival_at_time(model, X_scaled, time_horizon, batch_size=500):
    X_scaled = np.asarray(X_scaled, dtype="float32")

    predictions = np.empty(len(X_scaled), dtype=float)

    for start in range(0, len(X_scaled), batch_size):
        end = min(start + batch_size, len(X_scaled))
        surv_df = model.predict_surv_df(X_scaled[start:end])
        model_times = surv_df.index.to_numpy(dtype=float)

        time_index = np.searchsorted(model_times, time_horizon, side="right") - 1

        time_index = np.clip(time_index,0,len(model_times) - 1)

        predictions[start:end] = (surv_df.iloc[time_index].to_numpy(dtype=float))
    return predictions

def predict_coxph_survival_at_time(cph_model, X, time_horizon):
    surv_df = cph_model.predict_survival_function(X, times=[time_horizon])
    return surv_df.iloc[0].to_numpy(dtype=float)

def predict_rsf_survival_at_time(rsf_model, X, time_horizon):
    surv_functions = rsf_model.predict_survival_function(X)
    return np.asarray([function(time_horizon) for function in surv_functions],dtype=float)

