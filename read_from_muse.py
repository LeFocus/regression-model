from pylsl import resolve_byprop, StreamInlet
import time
from pynput import keyboard
import csv 
import os

# run muselsl stream in terminal
# install via pip
# requires python 3.8-3.10
print("Looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=2)
if not streams:
    raise RuntimeError("No EEG stream found")

inlet = StreamInlet(streams[0])
print("Connected to EEG stream.")

# --- CSV setup ---
csv_file = open("eeg_data.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "EEG_ch1", "EEG_ch2", "EEG_ch3", "EEG_ch4", "pupil_diameter"])  # header

array = []
keep_running = True


def on_press(key):
    """
    This function is called when any key is pressed.
    """
    global keep_running
    print(f"Key {key} pressed. Stopping loop.")
    keep_running = False

    # Returning False stops the listener
    return False


# Set up the listener in a separate thread
# 'on_press' is the function to call when a key is pressedr
listener = keyboard.Listener(on_press=on_press)
listener.start()

print("EEG logging started. Press any key to stop.")

# Temporary file for latest pupil diameter
pupil_file = "latest_pupil.txt"

while keep_running:
    sample, timestamp = inlet.pull_sample()
    
    # Replace invalid EEG values
    sample = [None if v == -1000 else v for v in sample]

    # --- Read latest pupil diameter ---
    pupil_diameter = None
    if os.path.exists(pupil_file):
        try:
            with open(pupil_file, "r") as f:
                pupil_diameter = float(f.read().strip())
        except:
            pupil_diameter = None

    # --- Save EEG + pupil to CSV ---
    csv_writer.writerow([timestamp] + sample + [pupil_diameter])

# --- Cleanup ---
csv_file.close()
listener.join()
print("EEG data saved to eeg_data.csv")