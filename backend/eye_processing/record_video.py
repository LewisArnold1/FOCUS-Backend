import cv2
import time
import subprocess
import csv
import threading
import os

'''
Set video and timestamp filenames.
Set video duration - 60s for actual vids, do 5-10s to test it works first

Run file - video will record then be played back after it is saved
Press q during playback to stop

'''

# change to your first name + test number x
VIDEO_FILENAME = "firstname_test_x.avi"
TIMESTAMP_FILENAME = "firstname_test_x_timestamps.t"

# Set to 60s for recording videos (can use 5-10s if you want to test its working)
VIDEO_DURATION = 5  


def record_video(video_filename, timestamp_filename, duration):
    """
    Records a video for the specified duration and saves it to a file.
    """
    cap = cv2.VideoCapture(0)  # Capture from the default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get frame width and height for saving video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    print("Recording video...")
    timestamps = []
    frames = []
    start_time = time.time()
   
    # Record Video & save timestamps 
    with open(timestamp_filename, "w") as f:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # Save frame
            frames.append(frame)

            # Save timestamp
            timestamp = time.time() - start_time
            timestamps.append(timestamp)
            f.write(f"{timestamp:.6f}\n")
    cap.release()

    if len(timestamps) < 2:
        print("Error: Not enough frames recorded.")
        return

    # Calculate fps of recorded video
    avg_fps = len(timestamps) / (timestamps[-1] - timestamps[0])

    # Save video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, avg_fps, (frame_width,frame_height))
    for frame in frames:
        out.write(frame)
    out.release()

    print(f"Video saved: {video_filename} ({timestamps[-1]:.2f} seconds, {avg_fps:.2f} FPS).")



def play_video(video_filename, timestamp_filename):
    timestamp_path = timestamp_filename

    # Load timestamps
    with open(timestamp_path, "r") as f:
        timestamps = [float(line.strip()) for line in f.readlines()]

    cap = cv2.VideoCapture(video_filename)
    frame_idx = 0

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    print("Playing back video...")
    start_time = time.time()  # Start reference time
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx>=len(timestamps):
            break

        # Show frame at current timestamp
        cv2.imshow('Frame', frame)

        # Wait until next timestamp
        if frame_idx > 0:
            wait_time = timestamps[frame_idx] - timestamps[frame_idx - 1]
            elapsed_time = time.time() - start_time
            sleep_time = max(0, wait_time - elapsed_time)  # Prevent negative sleep
            time.sleep(sleep_time)
        
        # Reset start time and increment frame counter
        start_time = time.time()
        frame_idx += 1

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    print(f"Video playback lasted {timestamps[frame_idx-1]} seconds.")

    cap.release()
    cv2.destroyAllWindows()

record_video(VIDEO_FILENAME,TIMESTAMP_FILENAME,VIDEO_DURATION)
play_video(VIDEO_FILENAME,TIMESTAMP_FILENAME)
