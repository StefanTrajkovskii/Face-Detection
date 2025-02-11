import cv2
import numpy as np
import datetime
import time
from threading import Thread

# --- Helper: Compute IoU between two bounding boxes ---
def compute_iou(boxA, boxB):
    # Each box is (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# --- Threaded video stream class ---
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

# --- Initialize DNN face detector (using CPU settings) ---
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

tracker = None  # Tracker instance
bbox = None     # Bounding box (x, y, w, h)
detect_interval = 20  # Run detection every 20 frames
frame_count = 0
iou_threshold = 0.5  # If new detection overlaps current tracker by 50% or more, keep the tracker

# Start the threaded video stream
vs = VideoStream(src=0).start()
prev_time = time.time()

while True:
    frame = vs.read()
    if frame is None:
        break

    # Resize frame for consistency and speed
    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # Calculate FPS
    curr_time = time.time()
    delta = curr_time - prev_time
    fps = 1.0 / delta if delta > 0 else 0
    prev_time = curr_time

    # On scheduled frames OR if tracker is lost, run detection
    run_detection = (tracker is None) or (frame_count % detect_interval == 0)
    if run_detection:
        # Prepare blob for DNN face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        h, w = frame.shape[:2]
        best_conf = 0
        new_box = None

        # Find best detection
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and confidence > best_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                new_box = (startX, startY, endX - startX, endY - startY)
                best_conf = confidence

        if new_box is not None:
            # If a tracker already exists, compare IoU to decide reinitialization
            if tracker is not None and bbox is not None:
                iou = compute_iou(bbox, new_box)
                # If overlap is high, keep existing tracker (and update bbox variable from tracker update)
                if iou < iou_threshold:
                    # Reinitialize tracker with the new detection
                    bbox = new_box
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, bbox)
                    cv2.putText(frame, "Reinitializing tracker", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # No tracker exists; initialize one.
                bbox = new_box
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, bbox)
                cv2.putText(frame, "Face detected, tracking started", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # If tracker exists, update it
    if tracker is not None:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w_box, h_box = map(int, bbox)
            center_x = x + w_box // 2
            center_y = y + h_box // 2
            radius = int(max(w_box, h_box) * 0.9)

            # Blurring using a Gaussian blur and a circular mask
            blurred_frame = cv2.GaussianBlur(frame, (49, 49), 15)
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (11, 11), 0)
            mask = mask.astype(float) / 255.0
            mask = cv2.merge([mask, mask, mask])
            frame = (mask * blurred_frame + (1 - mask) * frame).astype(np.uint8)

            # Draw a red question mark at the center
            question_mark = "?"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 3
            (text_width, text_height), baseline = cv2.getTextSize(question_mark, font, font_scale, thickness)
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            cv2.putText(frame, question_mark, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
            cv2.putText(frame, "Tracking & Blurred", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # If tracker update fails, remove tracker to trigger detection next loop.
            tracker = None
            cv2.putText(frame, "Lost tracking, re-detecting", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw FPS counter in the top-right corner
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Optimized Face Tracking, Blurring, and FPS", frame)
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
