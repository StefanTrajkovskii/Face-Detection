import cv2
import datetime

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam capture 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

tracker = None  # To hold the tracker instance
bbox = None     # Bounding box for the face

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # If no tracker exists, try detecting a face.
    if tracker is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) > 0:
            # For simplicity, select the first detected face.
            bbox = faces[0]  # (x, y, w, h)
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, tuple(bbox))
            cv2.putText(frame, "Face detected, tracking started", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Update the tracker and get the updated bounding box.
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # If tracking fails, reset the tracker to allow re-detection.
            tracker = None
            cv2.putText(frame, "Lost tracking, re-detecting", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Face Tracking', frame)

    # Press 's' to take a snapshot or 'q' to quit.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y")
        filename = f"snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved as {filename}!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
