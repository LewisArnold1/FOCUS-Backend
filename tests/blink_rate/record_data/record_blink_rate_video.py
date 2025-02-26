import cv2
import time
import csv
from datetime import datetime, timedelta
from process_eye_metrics import process_eye
from blinks import BlinkProcessor
import matplotlib.pyplot as plt

class BlinkRateRecorder:
    def __init__(self, capture_source=0):
        self.capture = cv2.VideoCapture(capture_source)
        if not self.capture.isOpened():
            raise ValueError("Failed to open the camera.")
        self.blink_processor = BlinkProcessor(eye_ar_thresh=0.24, eye_ar_consec_frames=2)
        self.blink_counts = []
        self.start_time = datetime.now()

    def record_blink_rate(self, duration=30):
        print("Recording blink rate...")

        while (datetime.now() - self.start_time).seconds < duration:
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to capture frame.")
                break

            timestamp_dt = datetime.now()

            # Resize frame for faster processing
            frame_resized = cv2.resize(frame, (640, 480))

            # Process eye detection and extract landmarks
            try:
                left_eye, right_eye, _, _, _, _ = process_eye(frame_resized, timestamp_dt)

                # Debugging: Check if eyes are detected correctly
                if left_eye is None or right_eye is None or isinstance(left_eye, int) or isinstance(right_eye, int):
                    print("Eye landmarks not detected properly. Skipping frame...")
                    continue  # Skip processing if eyes are not detected

                # Process blink detection
                blink_detected, ear = self.blink_processor.process_blink(left_eye, right_eye)
                print(f"EAR: {ear:.3f}, Blink Detected: {blink_detected}")

            except ValueError as e:
                print(f"Error unpacking process_eye return values: {e}")
                continue

            # Store the blink detected status with the timestamp
            self.blink_counts.append((timestamp_dt, blink_detected))

            # Add timestamp to the frame
            time_text = timestamp_dt.strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {time_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Add blink status to the frame
            if blink_detected == 1:
                cv2.putText(frame, "Blink Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Blink detected at {time_text}")

            # Display the video feed
            cv2.imshow("Blink Detection", frame)

            # Wait for a small duration to avoid overloading the CPU and camera
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Press 'q' to exit early

        # Release the camera and close OpenCV windows
        self.capture.release()
        cv2.destroyAllWindows()

        self.calculate_blink_rate()

    def calculate_blink_rate(self):
        # Calculate blink rate per minute
        blink_rates = []
        if not self.blink_counts:
            print("No blinks detected.")
            return []

        start_time = self.blink_counts[0][0]
        current_time = start_time
        minute_blink_count = 0
        blink_in_progress = False

        for timestamp, blink in self.blink_counts:
            if blink == 1:  # Blink detected
                if not blink_in_progress:
                    minute_blink_count += 1
                    blink_in_progress = True
            else:
                blink_in_progress = False

            # Check if we reached a new minute
            if timestamp >= current_time + timedelta(minutes=1):
                blink_rates.append(minute_blink_count)
                current_time += timedelta(minutes=1)
                minute_blink_count = 0

        # Append remaining blink count for the last minute
        if minute_blink_count > 0:
            blink_rates.append(minute_blink_count)

        # Print the blink rates per minute
        print("Blink rates per minute:", blink_rates)
        
        # Save the blink rates to a CSV file
        self.save_to_csv(blink_rates)

        # Plot the blink rates
        self.plot_blink_rates(blink_rates)

        return blink_rates

    def save_to_csv(self, blink_rates):
        # Save blink rates to CSV file
        with open("blink_rate_data.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Blink Rate"])
            for i, blink_rate in enumerate(blink_rates):
                timestamp = self.start_time + timedelta(minutes=i)
                writer.writerow([timestamp.strftime("%H:%M:%S"), blink_rate])

    def plot_blink_rates(self, blink_rates):
        # Plot Blink Rate Graph
        timestamps = [self.start_time + timedelta(minutes=i) for i in range(len(blink_rates))]
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, blink_rates, marker='o', linestyle='-', color='r', label="Blink Rate per Minute")
        plt.xlabel("Timestamp (HH:MM:SS)")
        plt.ylabel("Blink Rate (Blinks/Minute)")
        plt.title("Blink Rate Over Time")
        plt.xticks(rotation=45)
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    recorder = BlinkRateRecorder()
    recorder.record_blink_rate(duration=30)
