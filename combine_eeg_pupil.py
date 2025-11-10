import numpy as np
import mne
from scipy.integrate import simps
import pandas as pd
import os


WINDOW_DURATION = 4.0
SFREQ = 256.0
PUPIL_NORMALIZATION = "zscore"
DROP_NA_ROWS = True

# Use posterior channels for alpha/beta ratio to avoid frontal EMG
ALPHABETA_CHANNELS = ["TP9", "TP10"]  # <-- key change

# Keep analysis bands as-is for features
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}
# Use a narrower beta for the RATIO only (less EMG contamination)
BETA_FOR_RATIO = (13.0, 20.0)  # <-- key change

EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# Optional window rejection if high EMG: reject when narrow-beta rel power too high
REJECT_EMG_WINDOWS = True
EMG_REL_BETA_THRESH = 0.35  # tune: 0.30–0.45 works well for many setups

# Welch PSD settings for better alpha resolution (~0.5 Hz)
N_PER_SEG = int(SFREQ * 2.0)   # 2 s segments
N_OVERLAP = int(N_PER_SEG * 0.5)
N_FFT = int(SFREQ * 4.0)       # 4 s FFT for smoother spectrum

def load_eeg_to_mne(filepath):
    print(f"Loading EEG data from {filepath}...")
    df = pd.read_csv(filepath).dropna(subset=EEG_CHANNELS)
    data = df[EEG_CHANNELS].values.T * 1e-6  # μV -> V
    first_ts = df["timestamp"].iloc[0]

    info = mne.create_info(EEG_CHANNELS, SFREQ, ch_types=["eeg"] * len(EEG_CHANNELS))
    info.set_meas_date(pd.to_datetime(first_ts, unit="s", utc=True).to_pydatetime())

    raw = mne.io.RawArray(data, info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    # Average reference often strengthens posterior alpha contrast
    raw.set_eeg_reference("average", projection=False, verbose=False)  # <-- key change

    raw.filter(l_freq=1.0, h_freq=30.0, verbose=False)
    return raw

def load_pupil_data(filepath, normalize="zscore"):
    print(f"Loading pupil data from {filepath}...")
    df = pd.read_csv(filepath).dropna()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["pupil_raw"] = df[["left_pupil_diameter_px", "right_pupil_diameter_px"]].mean(axis=1)

    norm = normalize.lower()
    if norm == "zscore":
        mu, sd = df["pupil_raw"].mean(), df["pupil_raw"].std()
        df["pupil_norm"] = (df["pupil_raw"] - mu) / (sd if sd and sd > 0 else 1)
    elif norm == "minmax":
        mn, mx = df["pupil_raw"].min(), df["pupil_raw"].max()
        df["pupil_norm"] = (df["pupil_raw"] - mn) / (mx - mn) if mx > mn else 0.0
    elif norm == "relative":
        k = max(1, int(0.1 * len(df)))
        baseline = df["pupil_raw"].iloc[:k].mean()
        df["pupil_norm"] = df["pupil_raw"] / (baseline if baseline > 0 else 1.0)
    else:
        df["pupil_norm"] = df["pupil_raw"]
    return df[["timestamp", "pupil_raw", "pupil_norm"]]

def band_mean_power(psd, freqs, fmin, fmax):
    sel = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(sel):
        return np.nan
    return simps(psd[sel], freqs[sel]) / (fmax - fmin)

def combine_eeg_pupil_raw(directory):
    directory = directory
    eeg_path = os.path.join(directory, "eeg.csv")
    pupil_path = os.path.join(directory, "pupil.csv")

    print("Starting feature extraction...")
    raw = load_eeg_to_mne(eeg_path)
    pupil_df = load_pupil_data(pupil_path, normalize=PUPIL_NORMALIZATION)

    events = mne.make_fixed_length_events(raw, duration=WINDOW_DURATION)
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=WINDOW_DURATION,
        baseline=None, preload=True, picks="eeg", verbose=False
    )

    print(f"Created {len(epochs)} epochs of {WINDOW_DURATION} seconds each.")

    # PSD with explicit Welch params
    spectrum = epochs.compute_psd(
        method="welch", fmin=1.0, fmax=30.0, verbose=False,
        n_per_seg=N_PER_SEG, n_fft=N_FFT, n_overlap=N_OVERLAP
    )
    psd_data = spectrum.get_data()  # (n_epochs, n_channels, n_freqs)
    freqs = spectrum.freqs
    ch_names = epochs.ch_names

    features = []
    reject_count = 0

    for i in range(len(epochs)):
        start_time = pd.to_datetime(
            epochs.events[i, 0] / SFREQ + raw.info["meas_date"].timestamp(), unit="s"
        )
        end_time = start_time + pd.Timedelta(seconds=WINDOW_DURATION)

        row = {"window_id": i, "start_time": start_time}
        band_sums = {b: 0.0 for b in BANDS}
        total_power = 0.0

        # EEG features (mean band power per channel + totals)
        for j, ch in enumerate(ch_names):
            psd_ch = psd_data[i, j, :]
            for band, (fmin, fmax) in BANDS.items():
                mp = band_mean_power(psd_ch, freqs, fmin, fmax)
                row[f"{ch}_{band}_mean"] = mp
                band_sums[band] += mp
            total_power += simps(psd_ch, freqs)

        # Relative band power
        denom = total_power if total_power > 0 else np.nan
        for band in BANDS:
            row[f"rel_{band}"] = (band_sums[band] / denom) if denom and not np.isnan(denom) else np.nan

        # --- Alpha/Beta ratio computed from POSTERIOR channels only + narrow beta ---
        ia = [ch_names.index(c) for c in ALPHABETA_CHANNELS if c in ch_names]
        alpha_sum_post = 0.0
        beta_sum_post = 0.0
        fmin_a, fmax_a = BANDS["alpha"]
        fmin_b, fmax_b = BETA_FOR_RATIO  # narrow beta for ratio

        for idx in ia:
            psd_ch = psd_data[i, idx, :]
            alpha_sum_post += band_mean_power(psd_ch, freqs, fmin_a, fmax_a)
            beta_sum_post  += band_mean_power(psd_ch, freqs, fmin_b, fmax_b)

        # Optional EMG-based rejection: if narrow-beta rel power is too high, skip window
        if REJECT_EMG_WINDOWS:
            rel_narrow_beta = (beta_sum_post / (alpha_sum_post + beta_sum_post)) if (alpha_sum_post + beta_sum_post) > 0 else 1.0
            if rel_narrow_beta > EMG_REL_BETA_THRESH:
                reject_count += 1
                continue  # skip this epoch entirely

        row["overall_alpha_beta_ratio"] = (alpha_sum_post / beta_sum_post) if (beta_sum_post and beta_sum_post > 0) else np.nan
        row["log_alpha_beta_ratio"] = np.log(alpha_sum_post + 1e-10) - np.log(beta_sum_post + 1e-10)

        # Pupil features
        mask = (pupil_df["timestamp"] >= start_time) & (pupil_df["timestamp"] < end_time)
        pupil_win = pupil_df.loc[mask]
        row["mean_pupil_diameter_raw"]  = pupil_win["pupil_raw"].mean()  if not pupil_win.empty else np.nan
        row["mean_pupil_diameter_norm"] = pupil_win["pupil_norm"].mean() if not pupil_win.empty else np.nan

        features.append(row)

    df = pd.DataFrame(features)
    if DROP_NA_ROWS:
        df = df.dropna()

    df["label"] = 0

    out = os.path.join(directory, "combined.csv")
    df.to_csv(out, index=False)

    kept = len(df)
    if REJECT_EMG_WINDOWS:
        print(f"ℹ️ Rejected {reject_count} EMG-heavy windows based on narrow-beta dominance.")
    print(f"\n✅ Saved {kept} rows to {out}")
    return f"{directory}/combined.csv"
