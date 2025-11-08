import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

# --- 1. Constants ---
SAMPLE_RATE = 256  # Hz (Muse 2 is 256 Hz)
WINDOW_SECONDS = 5  # Use 5-second windows
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
TOTAL_SECONDS = 60  # Generate 60 seconds of sample data
N_SAMPLES = SAMPLE_RATE * TOTAL_SECONDS

# EEG bands
ALPHA_BAND = (8.0, 12.0)
BETA_BAND = (12.0, 30.0)
FREQ_BANDS = {'alpha': ALPHA_BAND, 'beta': BETA_BAND}

# Filtering constants
FILTER_LOW_CUT = 1.0  # Hz (Removes DC offset)
FILTER_HIGH_CUT = 50.0  # Hz (Removes 50/60Hz noise)
FILTER_ORDER = 5
ARTIFACT_THRESHOLD_UV = 100  # Reject windows with peak-to-peak > 100uV


# --- 2. Helper Functions ---

def create_sample_data(n_samples, dc_offset=850, sample_rate=256):
    """Generates a realistic-looking fake EEG signal."""
    t = np.arange(n_samples) / sample_rate
    alpha_wave = 5 * np.sin(2 * np.pi * 10 * t)
    beta_wave = 2 * np.sin(2 * np.pi * 20 * t)
    noise = 2 * np.random.randn(n_samples)
    raw_signal = dc_offset + alpha_wave + beta_wave + noise
    return raw_signal


def apply_bandpass(data, lowcut=FILTER_LOW_CUT, highcut=FILTER_HIGH_CUT, fs=SAMPLE_RATE, order=FILTER_ORDER):
    """Applies a Butterworth bandpass filter to the data."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def get_band_ratios(data, fs=SAMPLE_RATE, bands=FREQ_BANDS):
    """Calculates band power ratios from a window of data."""
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, 1 / fs)

    pos_mask = xf > 0
    xf = xf[pos_mask]
    yf_power = 2.0 / n * np.abs(yf[pos_mask]) ** 2  # Power spectrum

    band_powers = {}
    for band, (low, high) in bands.items():
        band_mask = (xf >= low) & (xf <= high)
        band_power = np.sum(yf_power[band_mask])
        band_powers[band] = band_power

    ratios = {}
    if band_powers.get('alpha', 0) > 0 and band_powers.get('beta', 0) > 0:
        total_power = band_powers['alpha'] + band_powers['beta']
        if total_power > 0:
            ratios['alpha_beta_ratio'] = band_powers['alpha'] / band_powers['beta']
        else:
            ratios['alpha_beta_ratio'] = np.nan
    else:
        ratios['alpha_beta_ratio'] = np.nan

    return ratios, band_powers


# --- 3. Main Processing Logic ---

print(f"Generating {TOTAL_SECONDS}s of sample data for 4 channels...")
sensors = ['TP9', 'AF7', 'AF8', 'TP10']
raw_data = {}
for sensor in sensors:
    raw_data[sensor] = create_sample_data(N_SAMPLES, dc_offset=np.random.randint(800, 900))

# ** Simulate a bad sensor (AF7) **
bad_start = 20 * SAMPLE_RATE  # 20s in
bad_end = 30 * SAMPLE_RATE  # 30s in
raw_data['AF7'][bad_start:bad_end] = -1000.0  # Use the -1000 sentinel value

results = []
n_windows = N_SAMPLES // WINDOW_SAMPLES

for i in range(n_windows):
    start_sample = i * WINDOW_SAMPLES
    end_sample = (i + 1) * WINDOW_SAMPLES

    for sensor in sensors:
        window_raw = raw_data[sensor][start_sample:end_sample]

        # --- 1. Calculate Average Voltage (from RAW data) ---
        avg_voltage = np.mean(window_raw)

        is_bad_connection = False
        is_artifact = False
        ratio = np.nan  # Default to NaN

        # --- 2. Check for Bad Connection (like -1000) ---
        if np.any(window_raw == -1000.0):
            is_bad_connection = True
        else:
            # --- 3. Filter the data ---
            window_filtered = apply_bandpass(window_raw)

            # --- 4. Check for Artifacts (on filtered data) ---
            peak_to_peak = np.max(window_filtered) - np.min(window_filtered)
            if peak_to_peak > ARTIFACT_THRESHOLD_UV:
                is_artifact = True
            else:
                # --- 5. Calculate Ratios (only if clean) ---
                ratios, _ = get_band_ratios(window_filtered)
                ratio = ratios.get('alpha_beta_ratio', np.nan)

        results.append({
            'window_id': i,
            'sensor': sensor,
            'avg_raw_voltage': avg_voltage,
            'alpha_beta_ratio': ratio,
            'is_bad_connection': is_bad_connection,
            'is_artifact': is_artifact
        })

# --- 4. Convert to DataFrame for easy handling ---
df_results = pd.DataFrame(results)
print("\n--- Processing Results ---")
print(df_results.to_string())

# --- 5. Plotting with Matplotlib ---
print("\nGenerating plot...")

# Set up the figure
plt.figure(figsize=(15, 7))

# Plot average raw voltage for each sensor
for sensor in sensors:
    # Select data for the current sensor
    sensor_df = df_results[df_results['sensor'] == sensor]

    # Plot its average voltage vs. window_id
    plt.plot(
        sensor_df['window_id'],
        sensor_df['avg_raw_voltage'],
        label=f'{sensor} (Raw Avg)',
        marker='o',
        linestyle='--'
    )

# Add plot titles and labels
plt.title('Average Raw Voltage per Window (Simulated)', fontsize=16)
plt.xlabel('Window ID', fontsize=12)
plt.ylabel('Average Raw Voltage (µV)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_filename = 'average_voltage_per_window.png'
plt.savefig(plot_filename)
plt.show()

print(f"Plot saved to {plot_filename}")