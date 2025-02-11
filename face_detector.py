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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Live Face Detector', frame)

    # s to snapshot and q to close
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("snapshot.jpg", frame)
        print("Snapshot saved!")
    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


# python face_detector.py - command to run