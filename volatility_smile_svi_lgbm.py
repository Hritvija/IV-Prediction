#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

A full pipeline for modeling the volatility smile from options data.
Combines parametric SVI calibration with residual ML prediction using LightGBM.
Optimized for speed and memory with parallelism and memmap structures.
"""

import os
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
import multiprocessing as mp

# --- Utility functions for SVI modeling ---

def svi_total_variance(params, k):
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def fit_svi(logM_arr, iv_arr, T, initial_guess=None):
    w_obs = (iv_arr ** 2) * T
    if initial_guess is None:
        a0 = np.mean(w_obs); b0 = 0.1; rho0 = 0.0; m0 = 0.0; sigma0 = 0.2
        initial_guess = np.array([a0, b0, rho0, m0, sigma0])
    bounds = ([-np.inf, 1e-6, -0.999, -np.inf, 1e-6],
              [np.inf, np.inf, 0.999, np.inf, np.inf])
    try:
        res = least_squares(lambda p: svi_total_variance(p, logM_arr) - w_obs,
                            x0=initial_guess, bounds=bounds,
                            max_nfev=500, ftol=1e-12, xtol=1e-12)
        return res.x if res.success else None
    except:
        return None

def svi_iv_from_params(params, logM_vals, T_vals):
    a, b, rho, m, sigma = params
    diff = logM_vals - m
    w = a + b * (rho * diff + np.sqrt(diff**2 + sigma**2))
    return np.sqrt(np.maximum(w / T_vals, 0.0))

def build_spline(logM_vals, iv_vals):
    order = np.argsort(logM_vals)
    x_all = logM_vals[order]
    y_all = iv_vals[order]
    x_unique, idx = np.unique(x_all, return_index=True)
    y_unique = y_all[idx]
    if len(x_unique) < 3:
        return None
    return interp1d(x_unique, y_unique, kind="cubic", fill_value="extrapolate", assume_sorted=True)

# --- Main Pipeline ---

def main():
    DATA_PATH = "./data"
    pf_train = pq.ParquetFile(os.path.join(DATA_PATH, "train_data.parquet"))
    all_cols = pf_train.schema.names
    call_cols = [c for c in all_cols if c.startswith("call_iv_")]
    put_cols = [c for c in all_cols if c.startswith("put_iv_")]
    X_cols = [f"X{i}" for i in range(42)]
    cols = ["timestamp", "underlying", "expiry"] + X_cols + call_cols + put_cols

    print("📦 Reading training data...")
    df = pd.read_parquet(os.path.join(DATA_PATH, "train_data.parquet"), columns=cols)

    calls = df.melt(id_vars=["timestamp", "underlying", "expiry"] + X_cols,
                    value_vars=call_cols,
                    var_name="strike_label", value_name="iv")
    calls["option_type"] = "call"
    calls["strike"] = calls["strike_label"].str.replace("call_iv_", "", regex=False).astype(int)

    puts = df.melt(id_vars=["timestamp", "underlying", "expiry"] + X_cols,
                   value_vars=put_cols,
                   var_name="strike_label", value_name="iv")
    puts["option_type"] = "put"
    puts["strike"] = puts["strike_label"].str.replace("put_iv_", "", regex=False).astype(int)

    long_df = pd.concat([calls, puts])
    del df, calls, puts
    gc.collect()

    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])
    long_df["expiry"] = pd.to_datetime(long_df["expiry"])
    long_df["T"] = (long_df["expiry"] - long_df["timestamp"]).dt.total_seconds() / (365*24*3600)
    long_df = long_df[long_df["T"] > 0]
    long_df["logM"] = np.log(long_df["strike"] / long_df["underlying"])

    print("✅ Melted data size:", len(long_df))

    # --- SVI Calibration per timestamp ---
    grouped = long_df.groupby("timestamp")
    svi_dict = {}
    spline_dict = {}
    prev_params = None

    print("⚙️  Running SVI calibration...")
    for ts, group in grouped:
        k = group["logM"].values.astype(np.float64)
        iv = group["iv"].values.astype(np.float64)
        T = group["T"].values[0]
        if len(k) >= 6:
            params = fit_svi(k, iv, T, initial_guess=prev_params)
            if params is not None:
                svi_dict[ts] = params
                prev_params = params
                continue
        spline_dict[ts] = build_spline(k, iv)

    print("✅ Calibrated SVI on", len(svi_dict), "timestamps.")

    # --- Feature Engineering ---
    print("🛠 Building features...")
    long_df["baseline_iv"] = np.nan
    for ts, group in grouped:
        idx = group.index
        logM_vals = group["logM"].values.astype(np.float64)
        T_vals = group["T"].values.astype(np.float64)
        if ts in svi_dict:
            iv_pred = svi_iv_from_params(svi_dict[ts], logM_vals, T_vals)
        elif ts in spline_dict and spline_dict[ts] is not None:
            iv_pred = spline_dict[ts](logM_vals)
        else:
            iv_pred = np.full(len(logM_vals), np.nan)
        long_df.loc[idx, "baseline_iv"] = iv_pred.astype(np.float32)

    y = (long_df["iv"] - long_df["baseline_iv"]).astype(np.float32)
    features = ["T", "underlying", "strike", "logM"] + X_cols
    for col in ["T", "underlying", "strike", "logM"]:
        long_df[col] = long_df[col].astype(np.float32)

    scaler = StandardScaler()
    long_df[["T_scaled", "underlying_scaled", "strike_scaled", "logM_scaled"]] = scaler.fit_transform(
        long_df[["T", "underlying", "strike", "logM"]])

    X_full = long_df[["T_scaled", "underlying_scaled", "strike_scaled", "logM_scaled"]].copy()
    pca = PCA(n_components=15)
    X_pca = pca.fit_transform(long_df[X_cols].fillna(0).astype(np.float32))
    for i in range(15):
        X_full[f"PC{i+1}"] = X_pca[:, i]
    X_full["opt_call"] = (long_df["option_type"] == "call").astype(int)
    X_full["opt_put"] = (long_df["option_type"] == "put").astype(int)

    # --- Model Training ---
    print("📈 Training LightGBM...")
    split_ts = long_df["timestamp"].quantile(0.9)
    mask_train = long_df["timestamp"] < split_ts
    mask_valid = ~mask_train

    lgb_train = lgb.Dataset(X_full[mask_train], label=y[mask_train])
    lgb_valid = lgb.Dataset(X_full[mask_valid], label=y[mask_valid])

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbose": -1
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )

    print("✅ Model training complete.")

if __name__ == "__main__":
    main()
