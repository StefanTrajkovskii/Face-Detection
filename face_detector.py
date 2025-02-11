import cv2
import numpy as np
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
            # Compute the center of the detected face region
            center_x = x + w // 2
            center_y = y + h // 2
            # Define a smaller radius (adjust the factor as needed)
            radius = int(max(w, h) * 0.8)
            
            # Create a blurred version of the entire frame
            blurred_frame = cv2.GaussianBlur(frame, (99, 99), 30)
            
            # Create a mask with the same dimensions as frame (grayscale)
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            # Draw a filled circle on the mask around the face's center with the computed radius
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            # Feather the mask's edges with another Gaussian blur
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            
            # Normalize the mask to a 0-1 range and convert to 3 channels
            mask = mask.astype(float) / 255.0
            mask = cv2.merge([mask, mask, mask])
            
            # Blend the blurred frame and the original frame using the mask.
            frame = (mask * blurred_frame + (1 - mask) * frame).astype(np.uint8)
            
            # Draw a red question mark at the center of the circle.
            question_mark = "?"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 3
            (text_width, text_height), baseline = cv2.getTextSize(question_mark, font, font_scale, thickness)
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            cv2.putText(frame, question_mark, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
            
        else:
            # If tracking fails, reset the tracker to allow re-detection.
            tracker = None
            cv2.putText(frame, "Lost tracking, re-detecting", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Face Tracking, Circular Blurring & Question Mark', frame)

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
