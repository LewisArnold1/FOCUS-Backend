import cv2
import dlib
import time
import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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
        A = dist.euclidean(eye[1], eye[5])  # Vertical distance between top and bottom
        B = dist.euclidean(eye[2], eye[4])  # Vertical distance between middle points
        C = dist.euclidean(eye[0], eye[3])  # Horizontal distance between left and right

        # Eye aspect ratio (EAR) formula
        ear = (A + B) / (2.0 * C)
        return ear

    def process_blink(self, left_eye, right_eye):
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Detect blink (0 = no blink, 1 = blink detected)
        blink_detected = 1 if ear < self.eye_ar_thresh else 0
        return blink_detected, ear

# Method to calculate blink rate per minute
def calculate_blink_rate(blink_timestamps):
    """
    Calculate blink rate per minute from blink timestamps.
    Consecutive 1's are treated as a single blink until a 0 is encountered.
    """
    if not blink_timestamps:
        return []

    blink_counts = [entry.blink_count for entry in blink_timestamps]
    blink_rates = []          # List to store blink counts per minute
    blink_in_progress = False # Flag to track ongoing blink events

    start_time = blink_timestamps[0].timestamp
    current_time = start_time
    minute_blink_count = 0

    for index, blink in enumerate(blink_counts):
        if blink == 1:
            if not blink_in_progress:
                minute_blink_count += 1
                blink_in_progress = True
        else:
            blink_in_progress = False

        if blink_timestamps[index].timestamp >= current_time + timedelta(minutes=1):
            blink_rates.append(minute_blink_count)
            current_time += timedelta(minutes=1)
            minute_blink_count = 0

    if minute_blink_count > 0:
        blink_rates.append(minute_blink_count)

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
csv_file = "Soniya_blinkrate_2_30_data.csv"
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Blink Detection"])

# Start capturing
cap = cv2.VideoCapture(0)
start_time = time.time()
blink_log = []
metrics = []

print("Recording blinks for 45 minutes...")

# Initialize the variables to calculate the rolling blink rate
rolling_blink_rate = 0
blink_count_per_second = 0
last_update_time = time.time()

while time.time() - start_time < 1800:  # Run for 45 minutes (2700 seconds)
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
        
        # Record blink detection (0 or 1)
        timestamp = datetime.now()
        blink_log.append((timestamp.strftime('%H:%M:%S'), blink_detected))

        # Store metrics to calculate blink rate later
        metrics.append(Metric(timestamp=datetime.now(), blink_count=blink_detected))

        # Save to CSV
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp.strftime('%H:%M:%S'), blink_detected])

    # Update the rolling blink rate every second
    if time.time() - last_update_time >= 1:
        # Calculate blink rate per minute from the last second
        blink_rate_per_minute = sum([blink for _, blink in blink_log[-60:]])  # Last 60 seconds
        rolling_blink_rate = blink_rate_per_minute
        last_update_time = time.time()

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_minutes, elapsed_seconds = divmod(int(elapsed_time), 60)
    elapsed_time_str = f"{elapsed_minutes:02}:{elapsed_seconds:02}"

    # Display blink detection, EAR, blink rate, and elapsed time
    cv2.putText(frame, f"Blink Detected: {blink_detected}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Blink Rate: {rolling_blink_rate:.2f} blinks/min", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Elapsed Time: {elapsed_time_str}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Blink Detection", frame)

    # Exit early if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Calculate blink rate per minute
blink_rates = calculate_blink_rate(metrics)

# Print results
print("Blink recording stopped. Data saved in 'Soniya_blinkrate_45m_data.csv'.")

# Plot Blink Rate Graph
timestamps, blink_counts = zip(*blink_log)
plt.figure(figsize=(10, 5))
plt.plot(timestamps, blink_counts, marker='o', linestyle='-', color='b', label="Blink Detection")
plt.xlabel("Timestamp (HH:MM:SS)")
plt.ylabel("Blink Detection (0 or 1)")
plt.title("Blink Detection Over Time")
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
