import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()



while True:
    ret, frame = cap.read() # Read a frame

    if not ret: # Check if frame was read successfully
        print("Error: Could not read frame.")
        break

    cv2.imshow('Webcam Feed', frame) # Display the frame

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
