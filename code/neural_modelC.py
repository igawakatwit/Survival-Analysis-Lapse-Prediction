from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
import torchtuples as tt
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from torch import nn
import numpy as np
from sksurv.metrics import brier_score, integrated_brier_score

def make_structured_y(df):
    return np.array(
        [(bool(e), t) for e, t in zip(df["event"], df["duration_years"])],
        dtype=[('event', 'bool'), ('time', 'float64')]
    )

def compute_coxtime_brier_score(model, X_eval, y_train, y_eval, times=None):
    surv_df = model.predict_surv_df(X_eval)
    if times is None:
        times = surv_df.index.values.astype(float)

    max_test_time = y_eval["time"].max()
    times = times[times < max_test_time]
    surv_matrix = surv_df.T.values
    surv_matrix = surv_matrix[:, :len(times)]

    _, scores = brier_score(
        y_train,
        y_eval,
        surv_matrix,
        times
    )

    return times, scores

def compute_coxtime_integrated_brier_score(model, X_eval, y_train, y_eval, times=None):
    surv_df = model.predict_surv_df(X_eval)

    if times is None:
        times = surv_df.index.values.astype(float)

    max_test_time = y_eval["time"].max()
    times = times[times < max_test_time]

    surv_matrix = surv_df.T.values
    surv_matrix = surv_matrix[:, :len(times)]

    ibs = integrated_brier_score(y_train, y_eval, surv_matrix, times)
    return ibs


def prepare_features(df, feature_cols, scaler=None):
    X = df[feature_cols].values.astype("float32")
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, scaler

def prepare_survival_labels(df):
    durations = df["duration_years"].values.astype("float32")
    events = df["event"].values.astype("int64")
    return durations, events


def train_coxtime(X_train, durations_train, events_train, X_val, durations_val, events_val):
    labtrans = CoxTime.label_transform()

    y_train = labtrans.fit_transform(durations_train, events_train)
    y_val = labtrans.transform(durations_val, events_val)

    in_features = X_train.shape[1]

    net = MLPVanillaCoxTime(
        in_features=in_features,
        num_nodes=[128, 64, 32],
        batch_norm=True,
        dropout=0.20,
        activation=nn.ReLU
    )

    model = CoxTime(net, tt.optim.Adam(1e-3), labtrans=labtrans)

    batch_size = 256
    epochs = 64
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
        val_data=(X_val, y_val)
    )

    model.compute_baseline_hazards()
    return model

def evaluate_coxtime(model, X_train, durations_train, events_train, X_val, durations_val, events_val, X_test, durations_test, events_test,train_sample_size=5000):
    sample_size = min(train_sample_size, len(X_train))

    rng = np.random.default_rng(42)
    train_indices = rng.choice(
        len(X_train),
        size=sample_size,
        replace=False
    )

    X_train_sample = X_train[train_indices]
    durations_train_sample = durations_train[train_indices]
    events_train_sample = events_train[train_indices]

    surv_train = model.predict_surv_df(X_train_sample)
    risk_train = -surv_train.mean(axis=0).values

    c_train = concordance_index_censored(
        events_train_sample.astype(bool),
        durations_train_sample,
        risk_train
    )[0]

    surv_val = model.predict_surv_df(X_val)
    risk_val = -surv_val.mean(axis=0).values

    c_val = concordance_index_censored(
        events_val.astype(bool),
        durations_val,
        risk_val
    )[0]

    del surv_val
    del risk_val

    surv_test = model.predict_surv_df(X_test)
    risk_test = -surv_test.mean(axis=0).values

    c_test = concordance_index_censored(
        events_test.astype(bool),
        durations_test,
        risk_test
    )[0]

    return c_train, c_val, c_test

def run_coxtime(train_df, val_df, test_df, feature_cols):
    X_train, scaler = prepare_features(train_df, feature_cols)
    X_val, _ = prepare_features(val_df, feature_cols, scaler=scaler)
    X_test, _ = prepare_features(test_df, feature_cols, scaler=scaler)

    durations_train, events_train = prepare_survival_labels(train_df)
    durations_val, events_val = prepare_survival_labels(val_df)
    durations_test, events_test = prepare_survival_labels(test_df)

    model = train_coxtime(X_train, durations_train, events_train,
                          X_val, durations_val, events_val)

    c_train, c_val, c_test = evaluate_coxtime(
        model,
        X_train,
        durations_train,
        events_train,
        X_val,
        durations_val,
        events_val,
        X_test,
        durations_test,
        events_test,
        train_sample_size=5000
    )

    print(f"Cox-Time Train C-index: {c_train:.4f}")
    print(f"Cox-Time Val   C-index: {c_val:.4f}")
    print(f"Cox-Time Test  C-index: {c_test:.4f}")

    y_train_struct = make_structured_y(train_df)
    y_test_struct = make_structured_y(test_df)

    times, brier_scores = compute_coxtime_brier_score(model, X_test, y_train_struct, y_test_struct)
    print("Mean Cox-Time Brier Score:", np.mean(brier_scores))

    ibs = compute_coxtime_integrated_brier_score(model, X_test, y_train_struct, y_test_struct)
    print("Cox-Time Integrated Brier Score:", ibs)

    return model, scaler, c_train, c_val, c_test

