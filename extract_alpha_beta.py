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


# go from (channels,samples) -> (samples, channels)
data_for_mne = data.T

ch_names = ['TP9', 'AF7', 'AF8', 'TP10', 'AUX1']
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'misc']
sfreq = 256 # muse sampling frequency

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage, on_missing='ignore') # 'ignore' skips AUX1


# Scale by 1e-6 to get Volts
raw = mne.io.RawArray(data_for_mne * 1e-6, info)

# --- 2. Apply the Bandpass Filter ---
# We filter from 1-30 Hz since we only care about bands up to Beta
raw.filter(l_freq=1.0, h_freq=30.0)

bands = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (12.0, 30.0)
}

spectrum = raw.compute_psd(method='welch', picks='eeg', fmin=1.0, fmax=30.0)

# 2. Extract the data and freqs from it
psd = spectrum.get_data()
freqs = spectrum.freqs

band_powers = {}
for band, (fmin, fmax) in bands.items():
    # Find the frequency indices corresponding to the band
    freq_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]

    # Select the PSD values for this band
    # psd[:, freq_indices] selects all channels, but only the frequencies in our band
    psd_band = psd[:, freq_indices]

    # Select the corresponding frequencies
    freqs_band = freqs[freq_indices]

    # Compute the absolute power by integrating the PSD using Simpson's rule
    # `axis=-1` integrates along the frequency axis
    band_power = simps(psd_band, freqs_band, axis=-1)

    band_powers[band] = band_power


eeg_ch_names = raw.copy().pick_types(eeg=True).ch_names

df_powers = pd.DataFrame(band_powers, index=eeg_ch_names)

if 'alpha' in df_powers and 'beta' in df_powers:
    df_powers['alpha_beta_ratio'] = df_powers['alpha'] / df_powers['beta']

print("\n--- Ratios ---")
print(df_powers)

print("--- MNE Raw Object Created ---")
print(raw)
print(raw.info)
