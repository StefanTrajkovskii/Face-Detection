import cv2 

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam capture 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Count the number of faces detected and overlay the count on the frame
    face_count = len(faces)
    cv2.putText(frame, f"Faces: {face_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Live Face Detector', frame)

    # Press 's' to save a snapshot, or 'q' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("snapshot.jpg", frame)
        print("Snapshot saved!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
