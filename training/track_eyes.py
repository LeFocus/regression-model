import cv2
import mediapipe as mp
import math
import csv

# Function to compute pupil diameter from iris landmarks
def compute_pupil_diameter(landmarks, indices, w, h):
    points = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    # Approximate diameter as max distance between points
    max_dist = max(math.hypot(x1-x2, y1-y1) for i, (x1, y1) in enumerate(points)
                                        for j, (x2, y2) in enumerate(points) if i != j)
    return max_dist

# Initialize MediaPipe Face Mesh    
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # refine_landmarks=True gives iris points

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# --- CSV setup ---
csv_file = open("eye_data.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "left_x", "left_y", "right_x", "right_y", "pupil_diameter"])  # header

print("Eye tracking started. Press 'q' to quit.")

# --- Iris landmark indices ---
LEFT_EYE_INDICES = [474, 475, 476, 477]
RIGHT_EYE_INDICES = [469, 470, 471, 472]

# --- Main loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    left_x = left_y = right_x = right_y = pupil_diameter = None

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Compute average coordinates for left and right iris
        left_x = sum([face_landmarks.landmark[i].x for i in LEFT_EYE_INDICES]) / len(LEFT_EYE_INDICES) * w
        left_y = sum([face_landmarks.landmark[i].y for i in LEFT_EYE_INDICES]) / len(LEFT_EYE_INDICES) * h
        right_x = sum([face_landmarks.landmark[i].x for i in RIGHT_EYE_INDICES]) / len(RIGHT_EYE_INDICES) * w
        right_y = sum([face_landmarks.landmark[i].y for i in RIGHT_EYE_INDICES]) / len(RIGHT_EYE_INDICES) * h

        # Compute average pupil diameter
        left_diameter = compute_pupil_diameter(face_landmarks.landmark, LEFT_EYE_INDICES, w, h)
        right_diameter = compute_pupil_diameter(face_landmarks.landmark, RIGHT_EYE_INDICES, w, h)
        pupil_diameter = (left_diameter + right_diameter) / 2

        # Draw iris points
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Log data with timestamp
    timestamp = time.time()
    csv_writer.writerow([timestamp, left_x, left_y, right_x, right_y, pupil_diameter])

    # Display the frame
    cv2.imshow('MediaPipe Iris Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
csv_file.close()
cap.release()
cv2.destroyAllWindows()
print("Eye data saved to eye_data.csv")