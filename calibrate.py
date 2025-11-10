"""
Create per-user calibration from two short feature CSVs (~30s each):
 - one for label 1 (focused)
 - one for label 0 (not focused)

Outputs:
 - JSON with prediction-level affine mapping (a,b) so mean0->0, mean1->1
 - Per-user feature means/stds (μ,σ) to apply z-score before global scaler
 - Small PKL copy (joblib) with same info for fast loading

Assumptions:
 - The two CSVs were produced by your feature extractor (same columns)
 - Your trained model bundle is a joblib with keys:
    {"booster", "imputer", "features", "normalizer"}  (as in your current code)
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any

def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

def _to_dmatrix(X: np.ndarray, feature_names):
    return xgb.DMatrix(X, feature_names=feature_names)

def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def _feature_mu_sigma(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].to_numpy()
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    # avoid zeros
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


# USER ID can be anything that identifies the user, email
def callibrate(USER_ID, FOCUS_CSV, UNFOCUS_CSV):
    # should be saved only temporairly,


    bundle = joblib.load("models/xgb_focus_reg.pkl")
    booster = bundle["booster"]
    imputer = bundle["imputer"]
    features = bundle["features"]
    normalizer = bundle.get("normalizer", None)

    # Load the two short calibration CSVs
    df1 = pd.read_csv(FOCUS_CSV)
    df0 = pd.read_csv(UNFOCUS_CSV)

    # Basic checks
    _require_cols(df1, features)
    _require_cols(df0, features)

    # Build normalized, imputed matrices using the SAME pipeline as training
    #    (normalizer may support group z + global robust)
    def transform_df(df_in: pd.DataFrame) -> np.ndarray:
        if normalizer is not None:
            X_norm_df = normalizer.transform_df(df_in)  # returns df with feature_cols normalized
            X_imp = imputer.transform(X_norm_df[features])
        else:
            X_imp = imputer.transform(df_in[features])
        return X_imp

    X1 = transform_df(df1)
    X0 = transform_df(df0)

    # Raw predictions on the two snippets
    d1 = _to_dmatrix(X1, features)
    d0 = _to_dmatrix(X0, features)
    p1_raw = booster.predict(d1)
    p0_raw = booster.predict(d0)

    m1 = float(np.mean(p1_raw)) if p1_raw.size else np.nan
    m0 = float(np.mean(p0_raw)) if p0_raw.size else np.nan
    n1 = int(len(p1_raw))
    n0 = int(len(p0_raw))

    # Build affine mapping so m0 -> 0 and m1 -> 1
    #    y_cal = clip( a * y_raw + b, 0, 1 )
    denom = (m1 - m0)
    if not np.isfinite(denom) or abs(denom) < 1e-6:
        # Degenerate case: means overlap; fall back to identity shift to center around 0.5
        a = 1.0
        b = -m0  # maps m0->0
        note = "degenerate_means; used fallback mapping"
    else:
        a = 1.0 / denom
        b = -m0 * a
        note = "ok"

    # Preview calibrated preds (for sanity)
    p1_cal = _clip01(a * p1_raw + b)
    p0_cal = _clip01(a * p0_raw + b)
    m1_cal = float(np.mean(p1_cal)) if p1_cal.size else np.nan
    m0_cal = float(np.mean(p0_cal)) if p0_cal.size else np.nan

    # 6) Feature-level per-user z-score stats from BOTH snippets (more stable)
    #    These can be applied BEFORE your global robust scaler for this user.
    both = pd.concat([df0, df1], ignore_index=True)
    mu, sd = _feature_mu_sigma(both, features)

    # 7) Save calibration artifacts
    stamp = int(time.time())
    calib = {
        "user_id": USER_ID,
        "timestamp": stamp,
        "bundle_path": "models/xgb_focus_reg.pkl",
        "features": features,
        # prediction-level affine
        "pred_map": {
            "a": a,
            "b": b,
            "mean_pred_label0": m0,
            "mean_pred_label1": m1,
            "mean_cal_label0": m0_cal,
            "mean_cal_label1": m1_cal,
            "n_label0": n0,
            "n_label1": n1,
            "note": note
        },
        # feature-level z for this user (apply before global robust scaler)
        "feature_stats": {
            "mu": mu.tolist(),
            "sd": sd.tolist()
        }
    }

    # base = f"user_calibration_{USER_ID}"
    # json_path = os.path.join(OUTPUT_DIR, base + ".json")
    # pkl_path  = os.path.join(OUTPUT_DIR, base + ".pkl")
    #
    # with open(json_path, "w") as f:
    #     json.dump(calib, f, indent=2)
    # joblib.dump(calib, pkl_path)

    # print("\n✅ Saved calibration:")
    # print(f"- {json_path}")
    # print(f"- {pkl_path}")
    # print(f"Means raw -> m0={m0:.4f}, m1={m1:.4f}  |  after map -> m0={m0_cal:.4f}, m1={m1_cal:.4f}")
    # print(f"a={a:.6f}, b={b:.6f}  (y_cal = clip(a*y_raw + b, 0, 1))")
    # print(calib)
    return calib
