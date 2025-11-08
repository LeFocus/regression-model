from pylsl import resolve_byprop, StreamInlet
import time

# run muselsl stream in terminal
# install via pip
# requires python 3.8-3.10


print("Looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=2)
if not streams:
    raise RuntimeError("No EEG stream found")

inlet = StreamInlet(streams[0])

print("Connected to EEG stream.")

while True:
    sample, timestamp = inlet.pull_sample()
    print(f"Timestamp: {timestamp}, Sample: {sample}")
    time.sleep(1)  # Simulate some processing time, should match up with eye tracking
