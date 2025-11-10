# file: run_focus_inference.py
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

def run_focus_inference(
    df_features: pd.DataFrame,
    calib,
    bundle_path: str = "models/xgb_focus_reg.pkl"
) -> dict:
    """
    Parameters
    ----------
    df_features : pd.DataFrame
        Parsed features for one or more windows. Must contain at least the columns used in training.
    calib
        Calibration weights (as produced by your calibration step), e.g.:
        {
          "feature_stats": {"mu": [...], "sd": [...]},
          "pred_map": {"a": float, "b": float}
        }
        Pass None to skip per-user z-score and affine calibration.
    bundle_path : str
        Path to the trained model bundle saved earlier
        (a joblib with keys {"booster","imputer","features","normalizer"}).

    Returns
    -------
    dict with:
      - "n": number of rows scored
      - "scores_raw": np.ndarray of uncalibrated scores in [0,1]
      - "scores_calibrated": np.ndarray of calibrated scores in [0,1] (or None if no calib)
      - "used_features": list of feature names used
    """
    # --- Load bundle ---
    bundle = joblib.load(bundle_path)
    booster: xgb.Booster = bundle["booster"]
    imputer = bundle["imputer"]
    features: list[str] = bundle["features"]
    normalizer = bundle.get("normalizer", None)
    global_scaler = getattr(normalizer, "global_scaler", None)

    X_df = pd.DataFrame({f: df_features[f] if f in df_features.columns else np.nan for f in features})

    if calib and "feature_stats" in calib:
        mu = np.asarray(calib["feature_stats"].get("mu", []), dtype=float)
        sd = np.asarray(calib["feature_stats"].get("sd", []), dtype=float)
        if mu.size == len(features) and sd.size == len(features):
            sd_safe = np.where(sd > 0, sd, 1.0)
            X_df[features] = (X_df[features].astype(float) - mu) / sd_safe

    # --- Then global robust scaler from training bundle (if present) ---
    X_vals = X_df[features].to_numpy(dtype=float)
    if global_scaler is not None:
        X_vals = global_scaler.transform(X_vals)

    # --- Impute missing like in training ---
    X_imp = imputer.transform(X_vals)

    # --- Predict with Booster ---
    dmat = xgb.DMatrix(X_imp, feature_names=features)
    y_raw = booster.predict(dmat)
    y_raw = np.clip(y_raw, 0.0, 1.0)

    # --- Optional affine calibration of predictions ---
    y_cal = None
    if calib and "pred_map" in calib:
        a = float(calib["pred_map"].get("a", 1.0))
        b = float(calib["pred_map"].get("b", 0.0))
        y_cal = np.clip(a * y_raw + b, 0.0, 1.0)

    return {
        "n": len(X_df),
        "scores_raw": y_raw,
        "scores_calibrated": y_cal,
        "used_features": features,
    }

# Example (commented):
# import pandas as pd, json
# df = pd.read_csv("some_parsed_features.csv")  # must contain the model feature columns
# calib = json.load(open("calibration/user_calibration_alice.json"))
# out = run_focus_inference(df, calib, "models/xgb_focus_reg.pkl")
# print(out["scores_calibrated"])
