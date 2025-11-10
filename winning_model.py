import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import RobustScaler

import xgboost as xgb

# =========================
# USER SETTINGS
# =========================
INPUT_CSVS = [
    "ben_focussed/combined_features_fixed.csv",
    "ben_half_focussed/combined_features_fixed.csv",
    "ben_relaxed/combined_features_fixed.csv",
]
OUTPUT_DIR = "dataset_split"      # where train.csv & test.csv go
MODEL_DIR = "models"              # where the model bundle is saved
MODEL_NAME = "xgb_focus_reg.pkl"
FEATURES_JSON = "xgb_reg_features.json"

TEST_SIZE = 0.20                   # 20% test
INTERNAL_VAL_SIZE = 0.10           # 10% of TRAIN for early stopping; auto-disabled if too small
RANDOM_STATE = 42
GROUP_BY_DATE = True               # use date(start_time) as fallback group
RETRAIN_ON_TRAIN_ALL = True        # retrain on full TRAIN (train+val) with best_n

# === NORMALIZATION SETTINGS ===
# Which column identifies a user or session for per-group normalization.
# Prefer a persistent user id if you have it; otherwise session_id; fallback to date(start_time).
GROUP_COL_CANDIDATES = ["user_id", "session_id", "start_time"]  # in priority order
NORMALIZE_MODE = "group_z_then_global_robust"  # or: "global_robust" or "none"

# Native XGBoost parameters (keys compatible with xgb.train)
XGB_NATIVE_PARAMS = dict(
    # (y hat - y)^2
    objective="reg:squarederror",
    eval_metric="rmse",
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=2,
    reg_lambda=1.0,   # 'lambda' alias below
    reg_alpha=0.0,    # 'alpha' alias below
    tree_method="hist",
    seed=RANDOM_STATE,
)
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 150


# =========================
# HELPERS
# =========================


def load_and_concat(paths):
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing CSV: {p}")
        df = pd.read_csv(p)
        if "label" not in df.columns:
            raise ValueError(f"{p} must include a numeric 'label' column (0, 0.5, 1).")
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(data)} rows from {len(paths)} CSVs.")
    return data


def build_groups(df):
    if "session_id" in df.columns:
        g = df["session_id"].astype(str)
    elif GROUP_BY_DATE and "start_time" in df.columns:
        st = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
        g = st.dt.date.astype(str).fillna("unknown")
    else:
        return None
    if pd.unique(g).size < 2:
        print("Only one unique group found; disabling group-aware split.")
        return None
    return g


def safe_split_train_test(df):
    """Try group-aware -> stratified -> random 80/20."""
    y = df["label"]
    groups = build_groups(df)

    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        idx_tr, idx_te = next(gss.split(df, groups=groups))
        print("Using group-aware split (session_id/date).")
        return idx_tr, idx_te

    uniq = np.sort(np.unique(y.dropna()))
    if set(np.round(uniq, 3)) <= {0.0, 0.5, 1.0} and len(uniq) > 1:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        idx_tr, idx_te = next(sss.split(df, y))
        print("Using stratified split on label {0, 0.5, 1}.")
        return idx_tr, idx_te

    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    print("Using random split.")
    return idx_tr, idx_te

"""
Helper function to print data split
"""
def describe_split(df, idx_train, idx_test, name="80/20"):
    def dist(rows):
        vc = rows["label"].value_counts(dropna=False).sort_index()
        return ", ".join([f"{k}: {int(v)}" for k, v in vc.items()])
    print(f"{name} — Train: {len(idx_train)} | Test: {len(idx_test)} | Total: {len(df)}")
    print("  Train labels:", dist(df.iloc[idx_train]))
    print("  Test  labels:", dist(df.iloc[idx_test]))


def select_features(df):
    drop_cols = [c for c in ["start_time", "window_id", "session_id"] if c in df.columns]
    feats_df = df.drop(columns=drop_cols, errors="ignore")
    num_cols = feats_df.select_dtypes(include=[np.number]).columns.tolist()
    if "label" not in feats_df.columns:
        raise ValueError("Data must contain 'label'.")
    feature_cols = [c for c in num_cols if c != "label"]
    if not feature_cols:
        raise ValueError("No numeric features found.")
    return feature_cols


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def safe_make_internal_val(X_train_imp, y_train, train_df):
    """Create a train/val split from TRAIN ONLY, or return None to disable ES."""
    n_train = len(X_train_imp)
    print(f"Internal split candidate — TRAIN samples: {n_train}")
    if n_train < 40 or INTERNAL_VAL_SIZE <= 0:
        print("ℹ️ Not enough samples for internal validation (or disabled); training without early stopping.")
        return None

    groups_train = build_groups(train_df)
    if groups_train is not None and pd.unique(groups_train).size >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=INTERNAL_VAL_SIZE, random_state=RANDOM_STATE)
        tr_idx, va_idx = next(gss.split(X_train_imp, groups=groups_train))
        print("Using group-aware internal validation.")
        return tr_idx, va_idx

    # Try stratified by label if feasible
    uniq = np.sort(np.unique(y_train))
    if set(np.round(uniq, 3)) <= {0.0, 0.5, 1.0} and len(uniq) > 1:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=INTERNAL_VAL_SIZE, random_state=RANDOM_STATE)
        tr_idx, va_idx = next(sss.split(X_train_imp, y_train))
        print("Using stratified internal validation.")
        return tr_idx, va_idx

    # Fallback random
    tr_idx, va_idx = train_test_split(
        np.arange(n_train), test_size=INTERNAL_VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    print("Using random internal validation.")
    return tr_idx, va_idx


def to_dmatrix(X, y=None, feature_names=None):
    return xgb.DMatrix(X, label=y, feature_names=feature_names)


def train_with_early_stopping_native(X_tr, y_tr, X_va, y_va, feature_names, params, num_boost_round, es_rounds):
    """Train with early stopping using native xgboost.train."""
    native = params.copy()
    # map sklearn-style keys if present
    if "reg_lambda" in native:
        native["lambda"] = native.pop("reg_lambda")
    if "reg_alpha" in native:
        native["alpha"] = native.pop("reg_alpha")

    dtrain = to_dmatrix(X_tr, y_tr, feature_names)
    dvalid = to_dmatrix(X_va, y_va, feature_names)
    booster = xgb.train(
        params=native,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=es_rounds,
        verbose_eval=False,
    )
    best_n = booster.best_iteration + 1
    return booster, best_n


def retrain_final_native(X_all_train, y_all_train, feature_names, params, num_boost_round):
    native = params.copy()
    if "reg_lambda" in native:
        native["lambda"] = native.pop("reg_lambda")
    if "reg_alpha" in native:
        native["alpha"] = native.pop("reg_alpha")

    dtrain_all = to_dmatrix(X_all_train, y_all_train, feature_names)
    booster = xgb.train(
        params=native,
        dtrain=dtrain_all,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    return booster


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) Load + combine
    df = load_and_concat(INPUT_CSVS)

    # 2) 80/20 split
    idx_tr, idx_te = safe_split_train_test(df)
    describe_split(df, idx_tr, idx_te, "80/20")

    train_df = df.iloc[idx_tr].reset_index(drop=True)
    test_df  = df.iloc[idx_te].reset_index(drop=True)

    # Save split CSVs
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path  = os.path.join(OUTPUT_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\n✅ Saved split CSVs:\n- {train_path}\n- {test_path}")

    # 3) Features
    feature_cols = select_features(train_df)
    X_train_full = train_df[feature_cols].copy()
    y_train_full = train_df["label"].astype(float).values
    X_test = test_df[feature_cols].copy()
    y_test = test_df["label"].astype(float).values

    # Impute NaNs
    imp = SimpleImputer(strategy="median")
    X_train_full_imp = imp.fit_transform(X_train_full)
    X_test_imp = imp.transform(X_test)

    # 4) Internal validation (safe)
    internal = safe_make_internal_val(X_train_full_imp, y_train_full, train_df)

    # 5) Train (native xgb with or without ES)
    if internal is None:
        best_n = NUM_BOOST_ROUND
        booster = retrain_final_native(
            X_train_full_imp, y_train_full, feature_cols, XGB_NATIVE_PARAMS, best_n
        )
        print(f"\nTrained without early stopping. num_boost_round={best_n}")
    else:
        tr_idx, va_idx = internal
        X_tr, X_va = X_train_full_imp[tr_idx], X_train_full_imp[va_idx]
        y_tr, y_va = y_train_full[tr_idx], y_train_full[va_idx]

        _, best_n = train_with_early_stopping_native(
            X_tr, y_tr, X_va, y_va,
            feature_names=feature_cols,
            params=XGB_NATIVE_PARAMS,
            num_boost_round=NUM_BOOST_ROUND,
            es_rounds=EARLY_STOPPING_ROUNDS
        )

        if RETRAIN_ON_TRAIN_ALL:
            booster = retrain_final_native(
                X_train_full_imp, y_train_full, feature_cols, XGB_NATIVE_PARAMS, best_n
            )
        else:
            # If you prefer to keep the early-stopped booster trained on (X_tr), re-train that path differently.
            booster = retrain_final_native(
                X_tr, y_tr, feature_cols, XGB_NATIVE_PARAMS, best_n
            )
        print(f"\nBest num_boost_round from early stopping: {best_n}")

    # 6) Evaluate on 20% TEST
    # ==== evaluate on TEST ====
    dtest = to_dmatrix(X_test_imp, feature_names=feature_cols)
    y_pred_test = clip01(booster.predict(dtest))

    # Backward-compatible RMSE
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    try:
        rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    # Backward-compatible Spearman (old SciPy returns tuple)
    from scipy.stats import spearmanr
    try:
        spear = spearmanr(y_test, y_pred_test).statistic
    except AttributeError:
        spear = spearmanr(y_test, y_pred_test)[0]

    print("\n=== Test Metrics (20% holdout) ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"Spearman ρ: {spear:.4f}")

    # 7) Save bundle + features
    bundle_path = os.path.join(MODEL_DIR, MODEL_NAME)
    joblib.dump({"booster": booster, "imputer": imp, "features": feature_cols}, bundle_path)
    print(f"\n✅ Saved model bundle to {bundle_path}")

    with open(os.path.join(MODEL_DIR, FEATURES_JSON), "w") as f:
        json.dump({"features": feature_cols}, f, indent=2)
    print(f"Saved feature list to {os.path.join(MODEL_DIR, FEATURES_JSON)}")

    # 8) Show top features (gain)
    # Because we passed feature_names into DMatrix, names are preserved.
    gain_importance = booster.get_score(importance_type="gain")
    top = sorted(gain_importance.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("\nTop 15 features (gain):")
    for name, val in top:
        print(f"{name}: {val:.6f}")

    # After y_pred_test is computed
    test_out = test_df.copy()
    test_out["pred"] = y_pred_test
    test_out.to_csv(os.path.join(OUTPUT_DIR, "test_with_preds.csv"), index=False)
    print("Wrote test_with_preds.csv")


if __name__ == "__main__":
    main()
