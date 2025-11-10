import os
import sys
import __main__  # <-- Import __main__ module
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from sklearn.preprocessing import RobustScaler  # Assuming RobustScaler is from sklearn


# =======================================================================
# --- FIX 2: PRE-EMPTIVE PLACEHOLDER FOR 'group_keys' ---
# Your NormalizerBundle.transform_df method calls a function named 'group_keys'.
# This will cause a NameError after the unpickling is fixed.
# You must import or define the *actual* 'group_keys' function here.
# =======================================================================
def group_keys(keys, is_date):
    """
    Placeholder for the 'group_keys' function.
    Replace this with your actual import or definition.
    """
    print("WARNING: Using placeholder 'group_keys'.")
    if is_date:
        try:
            return pd.to_datetime(keys).dt.date
        except Exception:
            return keys
    return keys


# =======================================================================
# --- FIX 1: EXACT CUSTOM CLASS DEFINITION ---
# This class definition is correct and necessary.
# =======================================================================

@dataclass
class NormalizerBundle:
    mode: str
    feature_cols: List[str]
    group_col: Optional[str]
    group_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]  # {group: (mean, std)}
    global_scaler: Optional[RobustScaler]

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].copy()

        if self.mode == "none":
            return X

        # Per-group z first (if available)
        if self.mode.startswith("group_z") and self.group_col is not None and self.group_stats:
            is_date = (self.group_col == "start_time")
            # This line will now find the placeholder 'group_keys' function
            keys = group_keys(df[self.group_col], is_date)
            Xz = X.copy()
            for g in keys.unique():
                idx = (keys == g)
                if g in self.group_stats:
                    mu, sd = self.group_stats[g]
                    sd_safe = np.where(sd > 0, sd, 1.0)
                    Xz.loc[idx, self.feature_cols] = (X.loc[idx, self.feature_cols] - mu) / sd_safe
            X = Xz

        # Global robust as fallback / second stage
        if self.mode in ("group_z_then_global_robust", "global_robust"):
            if self.global_scaler is None:
                raise RuntimeError("Global scaler missing in NormalizerBundle.")
            X[:] = self.global_scaler.transform(X.values)

        return X


# =======================================================================
# --- END CUSTOM CLASS DEFINITION ---
# =======================================================================


# Define the original default path for internal reference
DEFAULT_BUNDLE_PATH = "models/xgb_focus_reg.pkl"


def run_focus_inference(
        df_features: pd.DataFrame,
        calib,
        bundle_path: str = DEFAULT_BUNDLE_PATH
) -> dict:
    # ... (docstring) ...
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

    # --- PATH RESOLUTION (Retained for robustness) ---
    # ... (existing code) ...
    resolved_path = bundle_path

    if not os.path.exists(resolved_path):
        # ... (existing code) ...
        script_dir = os.path.dirname(os.path.abspath(__file__))
        attempt_1_path = os.path.join(script_dir, bundle_path)

        if os.path.exists(attempt_1_path):
            resolved_path = attempt_1_path
        else:
            # ... (existing code) ...
            cwd = os.getcwd()
            attempt_2_path = os.path.join(cwd, bundle_path)

            if os.path.exists(attempt_2_path):
                resolved_path = attempt_2_path
            else:
                # ... (existing code) ...
                raise FileNotFoundError(
                    f"Model bundle not found. Tried: "
                    f"1. {bundle_path} (relative to CWD or Script, failed) "
                    f"2. {attempt_1_path} (relative to script, failed) "
                    f"3. {attempt_2_path} (relative to process CWD, failed)"
                )

    bundle_path = resolved_path
    # --- END PATH RESOLUTION ---

    # =======================================================================
    # --- FIX 3: INJECT CLASS INTO __main__ FOR UNPICKLING ---
    # This maps the class 'NormalizerBundle' from *this* module to the
    # name '__main__.NormalizerBundle', which is what the unpickler is
    # looking for.
    # =======================================================================
    __main__.NormalizerBundle = NormalizerBundle
    # =======================================================================

    # --- Load bundle ---
    try:
        # This joblib.load() call should now succeed
        bundle = joblib.load(bundle_path)
    except Exception as e:
        # ... (existing code) ...
        raise RuntimeError(f"Failed to load model bundle from {bundle_path}: {e}")

    booster: xgb.Booster = bundle["booster"]
    # ... (existing code) ...
    imputer = bundle["imputer"]
    features: list[str] = bundle["features"]

    normalizer = bundle.get("normalizer", None)
    # ... (existing code) ...
    global_scaler = getattr(normalizer, "global_scaler", None)

    X_df = pd.DataFrame({f: df_features[f] if f in df_features.columns else np.nan for f in features})

    if calib and "feature_stats" in calib:
        # ... (existing code) ...
        mu = np.asarray(calib["feature_stats"].get("mu", []), dtype=float)
        sd = np.asarray(calib["feature_stats"].get("sd", []), dtype=float)
        if mu.size == len(features) and sd.size == len(features):
            # ... (existing code) ...
            sd_safe = np.where(sd > 0, sd, 1.0)
            X_df[features] = (X_df[features].astype(float) - mu) / sd_safe

    # --- Then global robust scaler from training bundle (if present) ---
    X_vals = X_df[features].to_numpy(dtype=float)
    # ... (existing code) ...
    if global_scaler is not None:
        X_vals = global_scaler.transform(X_vals)

    # --- Impute missing like in training ---
    X_imp = imputer.transform(X_vals)

    # --- Predict with Booster ---
    # ... (existing code) ...
    dmat = xgb.DMatrix(X_imp, feature_names=features)
    y_raw = booster.predict(dmat)
    y_raw = np.clip(y_raw, 0.0, 1.0)

    # --- Optional affine calibration of predictions ---
    # ... (existing code) ...
    y_cal = None
    if calib and "pred_map" in calib:
        # ... (existing code) ...
        a = float(calib["pred_map"].get("a", 1.0))
        b = float(calib["pred_map"].get("b", 0.0))
        y_cal = np.clip(a * y_raw + b, 0.0, 1.0)

    return {
        "n": len(X_df),
        # ... (existing code) ...
        "scores_raw": y_raw,
        "scores_calibrated": y_cal,
        "used_features": features,
    }