import numpy as np
from sksurv.metrics import brier_score
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score

def random_survival_forest(train_df, val_df, test_df, features):
    def make_y(df):
        df_local = df.copy()
        df_local["event"] = df_local["event"].astype(bool)
        return df_local[["event", "duration_years"]].rename(
            columns={"event": "event", "duration_years": "time"}
        ).to_records(index=False)

    y_train = make_y(train_df)
    y_val = make_y(val_df)
    y_test = make_y(test_df)

    X_train = train_df[features]
    X_val = val_df[features]
    X_test = test_df[features]

    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        max_depth=8,
        n_jobs=1,
        random_state=42
    )

    rsf.fit(X_train, y_train)

    print("RSF Train C-index:", rsf.score(X_train, y_train))
    print("RSF Val C-index:", rsf.score(X_val, y_val))
    print("RSF Test C-index:", rsf.score(X_test, y_test))
    rsf_time_grid = np.sort(np.unique(y_train["time"]))

    times, brier = compute_rsf_brier_score(rsf, X_test, y_train, y_test)
    print("Mean RSF Brier Score:", np.mean(brier))

    ibs = compute_rsf_integrated_brier_score(rsf, X_test, y_train, y_test)
    print("RSF Integrated Brier Score:", ibs)

    return rsf, X_val, y_val, X_test, y_test, rsf_time_grid

def compute_rsf_brier_score(rsf, X_eval, y_train, y_eval, times=None):
    if times is None:
        max_time = np.max(y_eval["time"])
        times = np.linspace(
            0.01,
            max_time - 1e-6,
            100
        )
    surv_funcs = rsf.predict_survival_function(X_eval)

    surv_probs = np.asarray([
        fn(times) for fn in surv_funcs
    ])

    _, scores = brier_score(
        y_train,
        y_eval,
        surv_probs,
        times
    )

    return times, scores

def compute_rsf_integrated_brier_score(rsf, X_eval, y_train, y_eval, times=None):
    if times is None:
        max_time = np.max(y_eval["time"])
        times = np.linspace(
            0.01,
            max_time - 1e-6,
            100
        )

    surv_funcs = rsf.predict_survival_function(X_eval)

    surv_probs = np.asarray([
        fn(times) for fn in surv_funcs
    ])

    ibs = integrated_brier_score(y_train, y_eval, surv_probs, times)
    return ibs