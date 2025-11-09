from pylsl import resolve_byprop, StreamInlet
import time
from pynput import keyboard

# run muselsl stream in terminal
# install via pip
# requires python 3.8-3.10


print("Looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=2)
if not streams:
    raise RuntimeError("No EEG stream found")

inlet = StreamInlet(streams[0])

print("Connected to EEG stream.")

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
# 'on_press' is the function to call when a key is pressed
listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Looping... Press any key to stop.")

# This is your main loop
while keep_running:
    sample, timestamp = inlet.pull_sample()
    for i in range(4):
        if sample[i] == -1000:
            sample[i] = None
    time.sleep(1/300)
    array.append(sample)

with open("data.txt", "w") as f:
    for sample in array:
        f.write(str(sample))
        f.write("\n")

listener.join()