import cv2
import numpy as np
import datetime
import time
from threading import Thread
import face_recognition

# --- Load known face "samurai.jpg" and compute its encoding ---
known_face_encodings = []
known_face_names = []

try:
    samurai_image = face_recognition.load_image_file("samurai.jpg")
    samurai_encoding = face_recognition.face_encodings(samurai_image)[0]
    known_face_encodings.append(samurai_encoding)
    known_face_names.append("Samurai")
except Exception as e:
    print("Could not load samurai.jpg:", e)

# --- Helper: Compute Intersection-over-Union (IoU) between two boxes ---
def compute_iou(boxA, boxB):
    # Each box is in the format (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# --- Threaded Video Stream for improved performance ---
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open webcam.")
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- Initialize DNN Face Detector (using CPU) ---
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

tracker = None   # Tracker instance
bbox = None      # Bounding box (x, y, w, h)
detect_interval = 20      # Run DNN detection every 20 frames
recognition_interval = 15 # Run facial recognition every 15 frames
frame_count = 0
iou_threshold = 0.5
recognized_name = "Unknown"

# Start the video stream
vs = VideoStream(src=0).start()
prev_time = time.time()

while True:
    frame = vs.read()
    if frame is None:
        break

    # Resize frame for consistency and speed; make a copy for recognition
    frame = cv2.resize(frame, (640, 480))
    orig_frame = frame.copy()  # Unmodified frame used for face_recognition
    frame_count += 1

    # --- FPS Calculation ---
    curr_time = time.time()
    delta = curr_time - prev_time
    fps = 1.0 / delta if delta > 0 else 0
    prev_time = curr_time

    # --- Run DNN Face Detection every detect_interval frames (or if tracker lost) ---
    if tracker is None or frame_count % detect_interval == 0:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        h, w = frame.shape[:2]
        best_conf = 0
        new_box = None

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and confidence > best_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                new_box = (startX, startY, endX - startX, endY - startY)
                best_conf = confidence

        if new_box is not None:
            if tracker is not None and bbox is not None:
                iou = compute_iou(bbox, new_box)
                if iou < iou_threshold:
                    bbox = new_box
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, bbox)
                    cv2.putText(frame, "Reinitializing tracker", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                bbox = new_box
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, bbox)
                cv2.putText(frame, "Face detected, tracking started", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Update Tracker and Process the Tracked Face ---
    if tracker is not None:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w_box, h_box = map(int, bbox)
            center_x = x + w_box // 2
            center_y = y + h_box // 2
            radius = int(max(w_box, h_box) * 0.9)

            # --- Run Facial Recognition at set intervals ---
            if frame_count % recognition_interval == 0:
                # Convert bbox to face_recognition format: (top, right, bottom, left)
                top = y
                right = x + w_box
                bottom = y + h_box
                left = x
                face_location = [(top, right, bottom, left)]
                encodings = face_recognition.face_encodings(orig_frame, face_location)
                if len(encodings) > 0:
                    face_encoding = encodings[0]
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            recognized_name = known_face_names[best_match_index]
                        else:
                            recognized_name = "Unknown"
                    else:
                        recognized_name = "Unknown"
                else:
                    recognized_name = "Unknown"

            # --- Apply Circular Blurring Effect on the Face Region ---
            blurred_frame = cv2.GaussianBlur(frame, (49, 49), 15)
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (11, 11), 0)
            mask = mask.astype(float) / 255.0
            mask = cv2.merge([mask, mask, mask])
            frame = (mask * blurred_frame + (1 - mask) * frame).astype(np.uint8)

            # --- Draw Red Question Mark at the Face Center ---
            question_mark = "?"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 3
            (qm_width, qm_height), baseline = cv2.getTextSize(question_mark, font, font_scale, thickness)
            qm_x = center_x - qm_width // 2
            qm_y = center_y + qm_height // 2
            cv2.putText(frame, question_mark, (qm_x, qm_y), font, font_scale, (0, 0, 255), thickness)


            # --- Draw the recognized name ("Samurai") centered below the face ---
            recog_font_scale = 1
            recog_thickness = 2
            (name_width, name_height), name_baseline = cv2.getTextSize(recognized_name, font, recog_font_scale, recog_thickness)
            recog_x = center_x - name_width // 2
            recog_y = center_y + radius + name_height + 10  # adjust offset as needed
            cv2.putText(frame, recognized_name, (recog_x, recog_y),
                        cv2.FONT_HERSHEY_SIMPLEX, recog_font_scale, (0, 0, 255), recog_thickness)
        else:
            tracker = None
            cv2.putText(frame, "Lost tracking, re-detecting", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- Draw FPS Counter in the Top-Right Corner ---
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Recognition, Tracking, Blurring, and FPS", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y")
        filename = f"snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved as {filename}!")
    elif key == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
