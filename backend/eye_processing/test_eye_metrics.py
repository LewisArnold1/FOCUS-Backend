import os
import sys
import cv2
import time
import csv
from datetime import datetime

# Adjust the path to ensure imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PARENT_DIR)

# Import the function to test
from eye_processing.eye_metrics.process_eye_metrics import process_eye


def record_video(output_file, duration=10, frame_width=640, frame_height=480, fps=30):
    """Records a video for the specified duration and saves it to output_file."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {output_file}")
    return output_file


def process_and_log_video(video_file, output_csv):
    """Process the recorded video, detect blinks, and log results in a CSV."""
    video = cv2.VideoCapture(video_file)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    eyes_closed = [0, 0]
    total_blinks = 0

    # Open CSV file for writing blink data
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Blink (1 for closed, 0 for open)"])

        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Call the process_eye function
            eye_closed, ear, pupil = process_eye(frame)
            eyes_closed.append(eye_closed)
            frame_count += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            # Write blink detection result to the CSV
            #writer.writerow([frame_count, int(eye_closed)])

            # Blink detection logic
            if eyes_closed[-1] == 1 and eyes_closed[-2] == 0:
                total_blinks += 1

            # Display processing results in the console
            print(1 if eye_closed else 0)
            writer.writerow([frame_count,1 if eye_closed else 0, timestamp])

    video.release()
    print(f"Blink results logged in {output_csv}")


if __name__ == '__main__':
    video_output = os.path.join(CURRENT_DIR, "recorded_video.avi")
    csv_output = os.path.join(CURRENT_DIR, "blink_log.csv")

    print("Recording video for 10 seconds...")
    video_file = record_video(video_output, duration=10)

    if video_file:
        print("Processing video and logging results...")
        process_and_log_video(video_file, csv_output)
