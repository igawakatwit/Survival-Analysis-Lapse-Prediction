from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from matplotlib import pyplot as plt
import numpy as np
from sksurv.metrics import brier_score
from sksurv.metrics import integrated_brier_score

#duplicate columns create perfect multicollinearity, this drops any columns that are exact copies of each other
def prepare_for_cox(df):
    df = df.copy()
    #drops constant columns (a column with only one unique value)
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        print("Dropping constant columns:", constant_cols)
        df = df.drop(columns=constant_cols)

    #duplicate columns create perfect multicollinearity which breaks the cox model
    X = df.drop(columns=["duration_years", "event"])
    #X.T transposes the dataframe, duplicated checks for identical columns which are removed
    dup_mask = X.T.duplicated()
    dup_cols = X.columns[dup_mask]
    if len(dup_cols) > 0:
        print("Dropping duplicate columns:", list(dup_cols))
        df = df.drop(columns=list(dup_cols))
    return df

#prepares the dataset, selects the model features, fits a penalized cox model, prints model summary
def cox_model(train_df, val_df, test_df):
    #clean the dataset before modeling
    train_df = prepare_for_cox(train_df)
    #select the feature columns, excluding duration_years (survival time), event (lapse = 1)
    features = [c for c in train_df.columns if c not in ["duration_years", "event"]]

    #initialize a penalized cox model
    #penalization stabilizes coefficients, reduces variance, prevents overfitting, helps convergence
    cph = CoxPHFitter(penalizer=1.0)

    #trains the model using engineered features, survival time, and event indicator
    cph.fit(
        train_df[features + ["duration_years", "event"]],
        duration_col="duration_years",
        event_col="event"
    )
    #prints hazard ratios
    hr = cph.hazard_ratios_
    print(hr.sort_values(ascending=False))

    print("Train C-index:", c_index(train_df, cph, features))
    print("Val C-index:", c_index(val_df, cph, features))
    print("Test C-index:", c_index(test_df, cph, features))


    return cph, features


def c_index(df, cph, features):
    df = df.copy()
    df = df[features + ["duration_years", "event"]]
    return concordance_index(
        df["duration_years"],
        #the cox model outputs a risk score (partial hazard), where higher score = higher predicted hazard = higher lapse risk
        #concordance_index expects higher predicted values = longer survival (lower risk), so negative to match
        -cph.predict_partial_hazard(df[features]),
        df["event"]
    )
def check_ph_assumptions(df, cph, show_plots=True, columns=None):
    #covariates actually used in the model
    model_cols = list(cph.params_.index)

    #strata variables used in the model
    strata_cols = []
    if hasattr(cph, "strata"):
        strata_cols = list(cph.strata)

    #build dataframe for PH test
    df_ph = df[model_cols + strata_cols + ["duration_cyears", "event"]].copy()

    #run PH test
    cph.check_assumptions(df_ph, p_value_threshold=0.05, show_plots=show_plots, columns=columns)

    if show_plots:
        plt.show()

def compute_brier_score(train_df, eval_df, cph, features, times=None):
    y_train = np.array(
        list(zip(train_df["event"].astype(bool),
                 train_df["duration_years"])),
        dtype=[("event", bool), ("time", float)]
    )

    y_eval = np.array(
        list(zip(eval_df["event"].astype(bool),
                 eval_df["duration_years"])),
        dtype=[("event", bool), ("time", float)]
    )

    if times is None:
        max_time = eval_df["duration_years"].max()

        times = np.linspace(0.01, max_time - 1e-6,100)

    surv_probs = cph.predict_survival_function(eval_df[features], times=times).T.values

    _, scores = brier_score(
        y_train,
        y_eval,
        surv_probs,
        times
    )

    return times, scores

def compute_integrated_brier_score(train_df, eval_df, cph, features, times=None):
    y_train = np.array(
        list(zip(train_df["event"].astype(bool),
                 train_df["duration_years"])),
        dtype=[("event", bool), ("time", float)]
    )

    y_eval = np.array(
        list(zip(eval_df["event"].astype(bool),
                 eval_df["duration_years"])),
        dtype=[("event", bool), ("time", float)]
    )

    if times is None:
        max_time = eval_df["duration_years"].max()

        times = np.linspace(0.01, max_time - 1e-6,100)

    surv_probs = cph.predict_survival_function(eval_df[features], times=times).T.values

    ibs = integrated_brier_score(y_train, y_eval, surv_probs, times)

    return ibs