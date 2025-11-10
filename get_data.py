import cv2
import mediapipe as mp
from pylsl import resolve_byprop, StreamInlet
import time
import threading
from pynput import keyboard
import math
import csv

# --- 1. Global Variables ---
eeg_data = []  # Will store [timestamp, TP9, AF7, AF8, TP10, AUX1]
pupil_data = []  # Will store [timestamp, left_diam_px, right_diam_px]
keep_running = True


# --- 2. Helper Function for Pupil Diameter ---q
def get_pixel_distance(lm1, lm2, w, h):
    """Calculates the pixel distance between two mediapipe landmarks."""
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# --- 3. pynput Key Listener (Thread 1) ---
def on_press(key):
    global keep_running
    print(f"Key {key} pressed.")  # Good for debugging

    try:
        # Check if the character of the key is 'q'
        if key.char == 'q':
            print("Stop key 'q' pressed. Stopping all streams.")
            keep_running = False
            return False  # Stop the listener thread

    except AttributeError:
        # This catches special keys (e.g., Shift, Ctrl, Esc)
        # that don't have a .char attribute
        pass

    # If it's any other key, the function ends (implicitly returning None),
    # which tells the listener to keep running.

listener = keyboard.Listener(on_press=on_press)
listener.start()


# --- 4. EEG Worker Function (Thread 2) ---
def collect_eeg(inlet):
    global keep_running, eeg_data
    print("EEG collection thread started.")
    while keep_running:
        sample, timestamp = inlet.pull_sample()

        # Process sample: replace -1000 with None
        processed_sample = []
        for i in range(5):  # Check all 5 channels
            processed_sample.append(None if sample[i] == -1000 else sample[i])

        # Store timestamp + sample data
        eeg_data.append([timestamp] + processed_sample)

        # Sleep to be "nice" to the CPU
        time.sleep(1 / 500)  # Sleep for a fraction of the sample rate
    print("EEG collection thread finished.")


# --- 5. Setup LSL Stream ---
print("Looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=2)
if not streams:
    raise RuntimeError("No EEG stream found. Is muselsl streaming?")

inlet = StreamInlet(streams[0])
print("Connected to EEG stream.")

# --- 6. Setup MediaPipe / OpenCV ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    keep_running = False

# --- 7. Start the EEG Thread ---
eeg_thread = threading.Thread(target=collect_eeg, args=(inlet,))
eeg_thread.start()

print("Main loop starting. Press any key to stop...")

# Define Iris landmark indices for vertical diameter
LEFT_IRIS_TOP = 474
LEFT_IRIS_BOTTOM = 476
RIGHT_IRIS_TOP = 469
RIGHT_IRIS_BOTTOM = 471

# --- 8. Main Thread (OpenCV/MediaPipe Loop) ---
try:
    while keep_running and cap.isOpened():
        # Get timestamp for this frame
        frame_timestamp = time.time()

        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        h, w, _ = frame.shape
        left_diam_px = None
        right_diam_px = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get all landmarks
                lm = face_landmarks.landmark

                # --- Calculate Pupil Diameter ---
                left_diam_px = get_pixel_distance(lm[LEFT_IRIS_TOP], lm[LEFT_IRIS_BOTTOM], w, h)
                right_diam_px = get_pixel_distance(lm[RIGHT_IRIS_TOP], lm[RIGHT_IRIS_BOTTOM], w, h)

                # Draw circles on iris (from your original script)
                iris_indices = [LEFT_IRIS_TOP, LEFT_IRIS_BOTTOM, RIGHT_IRIS_TOP, RIGHT_IRIS_BOTTOM]
                for idx in iris_indices:
                    x = int(lm[idx].x * w)
                    y = int(lm[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # Add data to our list
            pupil_data.append([frame_timestamp, left_diam_px, right_diam_px])

            # Print to console
            print(f"L_Pupil: {left_diam_px:.2f}px, R_Pupil: {right_diam_px:.2f}px")

        cv2.imshow('MediaPipe Iris Tracking', frame)
        cv2.waitKey(1)

finally:
    # --- 9. Cleanup ---
    print("Cleaning up resources...")

    eeg_thread.join()
    listener.join()
    cap.release()
    cv2.destroyAllWindows()

    # --- 10. Save Data to CSV Files ---

    # Write EEG data
    print(f"Saving {len(eeg_data)} EEG samples to eeg.csv...")
    with open("ben_relaxed/eeg.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'TP9', 'AF7', 'AF8', 'TP10', 'AUX1'])
        writer.writerows(eeg_data)

    # Write Pupil data
    print(f"Saving {len(pupil_data)} pupil samples to pupil.csv...")
    with open("ben_relaxed/pupil.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'left_pupil_diameter_px', 'right_pupil_diameter_px'])
        writer.writerows(pupil_data)

    print("Done.")