import numpy as np
import mne
from scipy.integrate import simps
import pandas as pd

data = []
with open("from_kiet.txt", "r") as f:
    for line in f:
        list_string = eval(line)
        if any(x is None for x in list_string):
            continue
        data.append(list_string)

data = np.array(data)
data_for_mne = data.T  # (channels, samples)

ch_names = ['TP9', 'AF7', 'AF8', 'TP10', 'AUX1']
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'misc']
sfreq = 256  # muse sampling frequency

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage, on_missing='ignore')

raw = mne.io.RawArray(data_for_mne * 1e-6, info)
raw.filter(l_freq=1.0, h_freq=30.0)

# --- 2. Create 10-Second Epochs ---
events = mne.make_fixed_length_events(raw, duration=1.0, overlap=0.0)
epochs = mne.Epochs(raw, events, tmin=0.0, tmax=1.0,
                    baseline=None, preload=True, picks='eeg')

print(f"\nCreated {len(epochs)} epochs of 10 seconds each.")

# --- 3. Compute PSD for each epoch ---
bands = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (12.0, 30.0)
}

spectrum = epochs.compute_psd(method='welch', fmin=1.0, fmax=30.0)
psd = spectrum.get_data()
freqs = spectrum.freqs
eeg_ch_names = epochs.ch_names

results = []
# Loop over each epoch
for i in range(len(epochs)):
    epoch_features = {'epoch_id': i}

    all_alpha_powers = []
    all_beta_powers = []

    # Loop over each EEG channel
    for j, ch_name in enumerate(eeg_ch_names):
        psd_epoch_channel = psd[i, j, :]
        channel_band_powers = []

        # Calculate power for each band
        for band, (fmin, fmax) in bands.items():
            freq_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
            psd_band = psd_epoch_channel[freq_indices]
            freqs_band = freqs[freq_indices]

            power = simps(psd_band, freqs_band, axis=-1)

            epoch_features[f'{ch_name}_{band}'] = power
            channel_band_powers.append(power)

            if band == 'alpha':
                all_alpha_powers.append(power)
            elif band == 'beta':
                all_beta_powers.append(power)

        # Calculate mean band power for this channel
        mean_band_power = np.mean(channel_band_powers)
        epoch_features[f'{ch_name}_mean_band_power'] = mean_band_power

        # ⚡ NEW: Calculate per-channel alpha/beta ratio
        alpha_power = epoch_features[f'{ch_name}_alpha']
        beta_power = epoch_features[f'{ch_name}_beta']

        # Add safety check for division by zero
        ratio = alpha_power / beta_power if beta_power > 0 else np.nan
        epoch_features[f'{ch_name}_alpha_beta_ratio'] = ratio

    # ⚡ NEW: Calculate the "overall" alpha/beta ratio for this epoch
    total_alpha = np.sum(all_alpha_powers)
    total_beta = np.sum(all_beta_powers)

    overall_ratio = total_alpha / total_beta if total_beta > 0 else np.nan
    epoch_features['overall_alpha_beta_ratio'] = overall_ratio

    results.append(epoch_features)

# Convert the final list of results into a DataFrame
df_results = pd.DataFrame(results)
pd.set_option('display.max_columns', None)  # Show all columns

print("\n--- Features for each 10-second Epoch (Window) ---")
print(df_results.head())

# You can save this DataFrame to a CSV
df_results.to_csv('my_epoch_features.csv', index=False)