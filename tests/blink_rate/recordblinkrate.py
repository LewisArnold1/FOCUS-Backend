import cv2
import dlib
import time
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import distance as dist
from collections import namedtuple

# BlinkProcessor class
class BlinkProcessor:
    def __init__(self, eye_ar_thresh=0.24, eye_ar_consec_frames=0):
        self.eye_ar_thresh = eye_ar_thresh
        self.eye_ar_consec_frames = eye_ar_consec_frames
        self.blink_detected = 0
        self.total_blinks = 0

    @staticmethod
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def process_blink(self, left_eye, right_eye):
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        blink_detected = 1 if ear < self.eye_ar_thresh else 0
        return blink_detected, ear

# Method to calculate blink rate per minute
def calculate_blink_rate_per_minute(metrics):
    blink_rates = []
    total_blinks = 0
    start_time = metrics[0].timestamp

    for metric in metrics:
        total_blinks += metric.blink_count
        duration_seconds = (metric.timestamp - start_time).total_seconds()
        if duration_seconds > 0:
            blink_rates.append(round((total_blinks * 60) / duration_seconds, 2))
        else:
            blink_rates.append(0)
    
    return blink_rates

# Define a namedtuple to store metrics
Metric = namedtuple("Metric", ["timestamp", "blink_count"])

# Load face detector & landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmarks
LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]

# Initialize blink processor
blink_processor = BlinkProcessor(eye_ar_thresh=0.24, eye_ar_consec_frames=2)

# Open CSV file to log blink rate
csv_file = "blink_rate_data.csv"
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Total Blinks"])

# Start capturing
cap = cv2.VideoCapture(0)
start_time = time.time()
blink_log = []
total_blinks = 0

print("Recording blinks for 10 minutes...")

metrics = []

while time.time() - start_time < 600:  # Run for 10 minutes (600 seconds)
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_IDX]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_IDX]

        blink_detected, ear = blink_processor.process_blink(left_eye, right_eye)
        
        if blink_detected:
            total_blinks += 1

    # Record timestamp & blink count
    timestamp = datetime.now().strftime('%H:%M:%S')
    blink_log.append((timestamp, total_blinks))

    # Save to CSV
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, total_blinks])

    # Store metrics to calculate blink rate later
    metrics.append(Metric(timestamp=datetime.now(), blink_count=total_blinks))

    # Display blinks & EAR on screen
    cv2.putText(frame, f"Total Blinks: {total_blinks}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Blink Detection", frame)

    # Exit early if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Calculate blink rate per minute
blink_rates = calculate_blink_rate_per_minute(metrics)

print("Blink recording stopped. Data saved in 'blink_rate_data.csv'.")

# Plot Blink Rate Graph
timestamps, blink_counts = zip(*blink_log)
plt.figure(figsize=(10, 5))
plt.plot(timestamps, blink_counts, marker='o', linestyle='-', color='b', label="Blink Count")
plt.xlabel("Timestamp (HH:MM:SS)")
plt.ylabel("Total Blinks")
plt.title("Blink Rate Over Time")
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.show()

# Plot Blink Rate Per Minute
plt.figure(figsize=(10, 5))
plt.plot([metric.timestamp for metric in metrics], blink_rates, marker='o', linestyle='-', color='r', label="Blink Rate per Minute")
plt.xlabel("Timestamp (HH:MM:SS)")
plt.ylabel("Blink Rate (Blinks/Minute)")
plt.title("Blink Rate Over Time")
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.show()
