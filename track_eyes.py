import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # refine_landmarks=True gives iris points

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Iris landmarks (for the first eye)
            left_eye_indices = [474, 475, 476, 477]
            right_eye_indices = [469, 470, 471, 472]

            h, w, _ = frame.shape
            for idx in left_eye_indices + right_eye_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # blue dots on pupils

    cv2.imshow('MediaPipe Iris Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
