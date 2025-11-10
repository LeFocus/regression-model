#!/usr/bin/env python3
import csv
import os
import joblib
from typing import List, Dict, Optional, Tuple
import aiofiles
import json
import pandas as pd

import calibrate
import combine_eeg_pupil
import run_inference
from fastapi import FastAPI, HTTPException, UploadFile, File, Form

# ============== CONFIG ==============
BUNDLE_PATH = os.environ.get("BUNDLE_PATH", "models/xgb_focus_reg.pkl")
CALIB_DIR   = os.environ.get("CALIB_DIR", "calibration")
os.makedirs(CALIB_DIR, exist_ok=True)

# Name of your extractor module + function.
# Change these two lines to match your actual API.
EXTRACTOR_MODULE = os.environ.get("EXTRACTOR_MODULE", "combine_eeg_pupil")
EXTRACTOR_FUNC   = os.environ.get("EXTRACTOR_FUNC", "extract_features_from_files")

UPLOAD_DIRECTORY = "raw_uploads"
BIAS_DIRECTORY = "bias"

# Ensure the upload directory exists when the application starts
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Expected signature:
#   df = extract_features_from_files(eeg_csv_path: str, pupil_csv_path: str) -> pandas.DataFrame
# The returned df must have the **same feature columns** used in training (plus any metadata like start_time, label, etc.)

ALLOW_UNCALIBRATED = True  # allow predictions even if no per-user calibration is saved

# ============== LOAD BUNDLE ==============
try:
    _bundle = joblib.load(BUNDLE_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model bundle at {BUNDLE_PATH}: {e}")

_booster = _bundle["booster"]            # xgboost.Booster
_imputer = _bundle["imputer"]            # SimpleImputer
_features: List[str] = _bundle["features"]
_normalizer = _bundle.get("normalizer", None)  # NormalizerBundle or None
_global_scaler = getattr(_normalizer, "global_scaler", None)

# ============== IMPORT EXTRACTOR ==============
try:
    extractor_mod = __import__(EXTRACTOR_MODULE, fromlist=[EXTRACTOR_FUNC])
    _extractor_func = getattr(extractor_mod, EXTRACTOR_FUNC)
except Exception as e:
    raise RuntimeError(
        f"Could not import {EXTRACTOR_FUNC} from {EXTRACTOR_MODULE}. "
        f"Set EXTRACTOR_MODULE/EXTRACTOR_FUNC env vars or adapt the code. Error: {e}"
    )

# ============== FASTAPI ==============
app = FastAPI(title="Focus Inference API (raw file version)", version="1.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "n_features": len(_features), "bundle": BUNDLE_PATH}

@app.post("/calibrate_raw")
async def calibrate_raw(user_id: str = Form(...),
    file1: UploadFile = File(..., description="eeg_Focus"),
    file2: UploadFile = File(..., description="eeg_Unfocussed"),
    file3: UploadFile = File(..., description="pupil_focussed"),
    file4: UploadFile = File(..., description="pupil_unfocused"),
    allow_uncalibrated: Optional[bool] = Form(None),

):
    """
    Uploads two CSV files and a user ID.
    Saves the files to the host server in the UPLOAD_DIRECTORY.
    """

    # 1. Validate file types
    if not file1.filename.endswith(".csv") or not file2.filename.endswith(".csv")\
        or not file3.filename.endswith(".csv") or not file4.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Both files must be .csv"
        )

    # We prepend the user_id to help organize files and avoid naming conflicts
    filepath1 = os.path.join(UPLOAD_DIRECTORY, f"{user_id}_{file1.filename}") # focussed
    filepath3 = os.path.join(UPLOAD_DIRECTORY, f"{user_id}_{file3.filename}")
    filepath2 = os.path.join(f"{UPLOAD_DIRECTORY}1", f"{user_id}_{file2.filename}") #unfocussed
    filepath4 = os.path.join(f"{UPLOAD_DIRECTORY}1", f"{user_id}_{file4.filename}")

    try:
        async with aiofiles.open(filepath1, 'wb') as f_out:
            while chunk := await file1.read(1024 * 1024):  # Read in 1MB chunks
                await f_out.write(chunk)

        # Save the second file (asynchronously in chunks)
        async with aiofiles.open(filepath2, 'wb') as f_out:
            while chunk := await file2.read(1024 * 1024):  # Read in 1MB chunks
                await f_out.write(chunk)

        async with aiofiles.open(filepath3, 'wb') as f_out:
            while chunk := await file3.read(1024 * 1024):  # Read in 1MB chunks
                await f_out.write(chunk)

        async with aiofiles.open(filepath4, 'wb') as f_out:
            while chunk := await file4.read(1024 * 1024):  # Read in 1MB chunks
                await f_out.write(chunk)

    except Exception as e:
        # If something goes wrong, return a server error
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while writing files: {e}"
        )
    finally:
        # Always close the file handles
        await file1.close()
        await file2.close()
        await file3.close()
        await file4.close()

    # use the data here
    data1 = combine_eeg_pupil.combine_eeg_pupil_raw(f"{UPLOAD_DIRECTORY}1")
    data2 = combine_eeg_pupil.combine_eeg_pupil_raw(UPLOAD_DIRECTORY)

    bias = calibrate.callibrate(user_id, f"{UPLOAD_DIRECTORY}/combined.csv",
                                f"{UPLOAD_DIRECTORY}1/combined.csv")
    with open(f"{BIAS_DIRECTORY}/{user_id}_bias", "w") as f_out:
        json.dump(bias, f_out) # dicstionary to json

    os.remove(filepath1) # delete after use
    os.remove(filepath2) # delete after use

    return {
        "message": "Bias's saved",
        "user_id": user_id
    }
@app.post("/predict_raw")
async def predict_raw(
    user_id: str = Form(...),
    eeg: UploadFile = File(..., description="CSV from Muse EEG"),
    pupil: UploadFile = File(..., description="CSV with pupil diameter"),
    allow_uncalibrated: Optional[bool] = Form(None),
):
    output = f"{user_id}"
    if not eeg.filename.endswith(".csv") or not pupil.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Both files must be .csv"
        )

    # We prepend the user_id to help organize files and avoid naming conflicts
    filepath1 = os.path.join(output, f"{user_id}_{eeg.filename}")
    filepath2 = os.path.join(output, f"{user_id}_{pupil.filename}")

    try:
        async with aiofiles.open(filepath1, 'wb') as f_out:
            while chunk := await eeg.read(1024 * 1024):  # Read in 1MB chunks
                await f_out.write(chunk)

        # Save the second file (asynchronously in chunks)
        async with aiofiles.open(filepath2, 'wb') as f_out:
            while chunk := await pupil.read(1024 * 1024):  # Read in 1MB chunks
                await f_out.write(chunk)

    except Exception as e:
        # If something goes wrong, return a server error
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while writing files: {e}"
        )
    finally:
        # Always close the file handles
        await eeg.close()
        await pupil.close()

    # use the files here
    combine_eeg_pupil.combine_eeg_pupil_raw(output)

    df = pd.read_csv(f"{output}/combined.csv")
    try:
        with open(f'{BIAS_DIRECTORY}/{user_id}_bias', 'r') as file:
            calibration_data = json.load(file)
    except Exception as e:
        calibration_data = None

    values = run_inference.run_focus_inference(df, calibration_data)

    if calibration_data:
        scores = values["scores"]
    else:
        scores = None

    total = 0
    for item in scores:
        total += item

    return total/len(scores) # return one number over the whole period