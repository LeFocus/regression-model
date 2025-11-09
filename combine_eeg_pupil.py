import numpy as np
import mne
from scipy.integrate import simps
import pandas as pd

# --- 1. Define Constants ---
WINDOW_DURATION = 2.0  # seconds
SFREQ = 256.0  # Muse 2 sampling frequency

# EEG bands
BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (12.0, 30.0)
}
EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']  # Channels to process


def load_eeg_to_mne(filepath):
    """Loads the eeg.csv file into an MNE Raw object."""
    print(f"Loading EEG data from {filepath}...")
    df = pd.read_csv(filepath)

    # Drop rows with any bad data from the sensors
    df = df.dropna(subset=EEG_CHANNELS)

    # Get the raw data portion and scale it (uV -> V)
    # MNE expects (channels, samples)
    data_for_mne = df[EEG_CHANNELS].values.T * 1e-6

    # Get the first timestamp to set the measurement date
    first_timestamp = df['timestamp'].iloc[0]

    # Create the MNE info object
    ch_types = ['eeg'] * len(EEG_CHANNELS)
    info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=SFREQ, ch_types=ch_types)
    info.set_meas_date(pd.to_datetime(first_timestamp, unit='s', utc=True).to_pydatetime())

    # Create the MNE Raw object
    raw = mne.io.RawArray(data_for_mne, info)

    # Set the 'montage' (sensor locations) for 3D plotting
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # Filter the data
    raw.filter(l_freq=1.0, h_freq=30.0, verbose=False)

    return raw


def load_pupil_data(filepath):
    """Loads and cleans the pupil.csv file."""
    print(f"Loading pupil data from {filepath}...")
    df = pd.read_csv(filepath)

    # Convert timestamps to datetime objects, which makes time-based selection easy
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Drop any rows where pupil couldn't be found
    df = df.dropna()

    # Calculate a single 'pupil_diameter' by averaging left and right
    df['pupil_diameter'] = df[['left_pupil_diameter_px', 'right_pupil_diameter_px']].mean(axis=1)

    return df[['timestamp', 'pupil_diameter']]


# --- 2. Main Processing ---
print("Starting feature extraction...")

# Load both datasets
raw_eeg = load_eeg_to_mne('eeg.csv')
pupil_df = load_pupil_data('pupil.csv')

# Create 10-second epochs from the EEG data
events = mne.make_fixed_length_events(raw_eeg, duration=WINDOW_DURATION, overlap=0.0)
epochs = mne.Epochs(raw_eeg, events, tmin=0.0, tmax=WINDOW_DURATION,
                    baseline=None, preload=True, picks='eeg', verbose=False)

print(f"Created {len(epochs)} epochs of {WINDOW_DURATION} seconds each.")

# Get the PSD (Power Spectral Density) for all epochs at once
spectrum = epochs.compute_psd(method='welch', fmin=1.0, fmax=30.0, verbose=False)
psd_data = spectrum.get_data()  # Shape: (n_epochs, n_channels, n_frequencies)
freqs = spectrum.freqs
eeg_ch_names = epochs.ch_names

# This will hold the feature row for each epoch
all_features = []

# Loop Through Epochs and Combine Features
for i in range(len(epochs)):
    # Get the start and end time of this specific epoch
    # We get the event time (in seconds) and add the raw object's start time
    epoch_start_time = pd.to_datetime(epochs.events[i, 0] / SFREQ + raw_eeg.info['meas_date'].timestamp(), unit='s')
    epoch_end_time = epoch_start_time + pd.Timedelta(seconds=WINDOW_DURATION)

    # This dictionary will hold all features for this one window
    feature_row = {
        'window_id': i,
        'start_time': epoch_start_time,
    }

    # --- A: Calculate EEG Features (Band Powers) ---
    all_band_powers = []
    for j, ch_name in enumerate(eeg_ch_names):
        psd_epoch_channel = psd_data[i, j, :]

        for band, (fmin, fmax) in BANDS.items():
            freq_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
            psd_band = psd_epoch_channel[freq_indices]
            freqs_band = freqs[freq_indices]

            power = simps(psd_band, freqs_band, axis=-1)
            feature_row[f'{ch_name}_{band}'] = power
            all_band_powers.append(power)

    # Add overall alpha/beta ratio
    alpha_sum = sum(feature_row[f'{ch}_alpha'] for ch in eeg_ch_names)
    beta_sum = sum(feature_row[f'{ch}_beta'] for ch in eeg_ch_names)
    feature_row['overall_alpha_beta_ratio'] = alpha_sum / beta_sum if beta_sum > 0 else np.nan

    # --- B: Calculate Pupil Features (Mean Diameter) ---

    # Select all pupil readings that fall within this epoch's time window
    mask = (pupil_df['timestamp'] >= epoch_start_time) & (pupil_df['timestamp'] < epoch_end_time)
    pupil_data_in_window = pupil_df.loc[mask]

    if not pupil_data_in_window.empty:
        # Calculate the average pupil diameter for this window
        mean_pupil = pupil_data_in_window['pupil_diameter'].mean()
        feature_row['mean_pupil_diameter'] = mean_pupil
    else:
        # No pupil data was found for this window
        feature_row['mean_pupil_diameter'] = np.nan

    # Add the completed feature row to our list
    all_features.append(feature_row)

# --- 4. Save Final Combined Feature CSV ---
print("Feature extraction complete.")

# Convert the list of features into a final DataFrame
df_features = pd.DataFrame(all_features)
df_features = df_features.dropna()  # Drop any windows that had missing data

print(f"\n--- Combined Features (first 5 rows) ---")
print(df_features.head())

output_filename = 'combined_features.csv'
df_features.to_csv(output_filename, index=False)
print(f"\nSuccessfully saved combined features to {output_filename}")