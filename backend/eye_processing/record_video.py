import cv2
import time
import json
from datetime import datetime
import os

'''
Set video and timestamp filenames.
Set video duration - 60s for actual vids, do 5-10s to test it works first

Run file - video will record then be played back after it is saved
Press q during playback to stop

'''

# change to your first name + test number x
VIDEO_FILENAME = "firstname_test_x.avi"
TIMESTAMP_FILENAME = "firstname_test_x_timestamps.txt"

# Set to 60s for recording videos (can use 5-10s if you want to test its working)
VIDEO_DURATION = 5


def record_video(video_filename, timestamp_filename, duration):
    """
    Records a video for the specified duration and saves it to a file.
    """

    # Current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Full paths
    video_path = os.path.join(script_dir, video_filename)   
    timestamp_path = os.path.join(script_dir, timestamp_filename)
    
    cap = cv2.VideoCapture(0)  # Capture from the default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get frame width and height for saving video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    timestamps = []
    frames = []

    print("Recording video...")
    start_time = datetime.now()

    # Store frames & timestamps locally
    while (datetime.now() - start_time).total_seconds() < duration:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Save frame & timestamp locally
        timestamps.append(datetime.now())
        frames.append(frame)

    cap.release()

    # Calculate avg FPS
    if len(timestamps) >= 2:
        avg_fps = len(timestamps) / (timestamps[-1] - start_time).total_seconds()
    else:
        print("Not enough frames recorded")
        return
    
    # Save video with avg FPS
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, avg_fps, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    out.release()

    # change timestamps to string for JSON compatibility
    timestamps_str = [timestamp.strftime('%Y-%m-%d %H:%M:%S.%f') for timestamp in timestamps]
        
    # Save data to json
    with open(timestamp_path, "w") as json_file:
        json.dump(timestamps_str, json_file, indent=4)
    
    elapsed_time = (timestamps[-1] - start_time).total_seconds()
    print(f"Video saved: {video_filename} ({elapsed_time:.2f} seconds, {avg_fps:.2f} FPS).")

def play_video(video_filename, timestamp_filename):
    video = video_filename
    timestamp_path = timestamp_filename

    '''
    # Load timestamps
    with open(timestamp_path, "r") as f:
        timestamps_str = [line.strip() for line in f.readlines()]

        timestamps = []

        # Remove the leading and trailing quotes
        timestamps_cleaned = [timestamp.strip('"') for timestamp in timestamps_str]

        
        # Optionally, convert to datetime objects for easier processing
        from datetime import datetime
        timestamps_dt = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") for ts in timestamps_cleaned]

        # Display cleaned timestamps
        for ts in timestamps_dt:
            print(ts)  
    '''
    cap = cv2.VideoCapture(video_filename)
    frame_idx = 0

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    print("Playing back video...")
    start_time = datetime.now()  # Start reference time
    
    while cap.isOpened():
        ret, frame = cap.read()
        # if not ret or frame_idx>=len(timestamps):
        #     break

        # Show frame at current timestamp
        cv2.imshow('Frame', frame)
        '''

        # Wait until next timestamp
        if frame_idx > 0:
            wait_time = timestamps[frame_idx] - timestamps[frame_idx - 1]
            elapsed_time = time.time() - start_time
            sleep_time = max(0, wait_time - elapsed_time)  # Prevent negative sleep
            time.sleep(sleep_time)
        
        # Reset start time and increment frame counter
        start_time = time.time()
        frame_idx += 1
        '''
        time.sleep(0.1)
        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # print(f"Video playback lasted {timestamps[frame_idx-1]} seconds.")

    cap.release()
    cv2.destroyAllWindows()

record_video(VIDEO_FILENAME,TIMESTAMP_FILENAME,VIDEO_DURATION)
# play_video(VIDEO_FILENAME,TIMESTAMP_FILENAME)
