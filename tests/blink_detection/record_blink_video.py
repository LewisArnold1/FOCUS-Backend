import cv2
import json
from datetime import datetime
import os
import time

# Import face processor to check eye is found in each frme
try:
    from eyes_closed_tests.face import FaceProcessor
except ImportError:
    from eyes_closed_tests.face import FaceProcessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'eyes_closed_tests','shape_predictor_68_face_landmarks.dat')
face_processor = FaceProcessor(PREDICTOR_PATH)

'''
Set video and timestamp filenames.
Set video duration - 60s for actual vids, do 5-10s to test it works first

Run file - video will record then be played back after it is saved
'''

# # change to your first name + test number x
VIDEO_FILENAME = "firstname_test_x.avi"
TIMESTAMP_FILENAME = "firstname_test_x_timestamps.txt"

VIDEO_FILENAME = "soniya_test_3.avi"
TIMESTAMP_FILENAME = "soniya_test_3_timestamps.txt"

# Set to 60s for recording videos (can use 5-10s if you want to test its working)
VIDEO_DURATION = 60


def record_video(video_filename, timestamp_filename, duration):
    """
    Records a video for the specified duration and saves it to a file.
    """

    # Current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Tests folder
    tests_dir = os.path.join(script_dir, "blink_test_files")

    # Full paths
    video_path = os.path.join(tests_dir, video_filename)   
    timestamp_path = os.path.join(tests_dir, timestamp_filename)
    
    cap = cv2.VideoCapture(0)  # Capture from the default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get frame width and height for saving video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    timestamps = []
    frames = []
    dropped_frames = 0

    print("Recording video...")
    start_time = datetime.now()

    # Store frames & timestamps locally
    while (datetime.now() - start_time).total_seconds() < duration:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # process_face to check eye is found
        _, left_eye, right_eye, _ = face_processor.process_face(frame)
        if left_eye is None or right_eye is None:
            dropped_frames+=1
            print(f"No eye no.{dropped_frames}\nat time {datetime.now().hour}:{datetime.now().minute}:{datetime.now().second:.2f}.")
        # Continue recording regardless

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

    # Current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Tests folder
    tests_dir = os.path.join(script_dir, "blink_test_files")

    # Full paths
    video_path = os.path.join(tests_dir, video_filename)   
    timestamp_path = os.path.join(tests_dir, timestamp_filename)

    # Load timestamps
    if os.path.exists(timestamp_path):  # Check if the file exists
        with open(timestamp_path, "r") as json_file:
            timestamps_str = json.load(json_file)  # Load JSON data
        timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') for ts in timestamps_str]  # Convert to datetime
    else:
        print("Timestamps file not found.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    elif len(timestamps) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        print("Timestamps or frames missing.")
        return
    else:
        print(f"Video has {len(timestamps)} frames/timestamps")        

    print("Playing back video...")
    start_time = datetime.now()
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()

        # process_face to check eye is found
        _, left_eye, right_eye, _ = face_processor.process_face(frame)
        if left_eye is None or right_eye is None:
            dropped_frames+=1
            print(f"No eye no.{dropped_frames}\nat time {datetime.now().hour}:{datetime.now().minute}:{datetime.now().second:.2f}.")
        # Continue regardless

        # Stop at last frame
        if not ret or frame_idx >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            break

        # Calculate relative timestamp
        relative_timestamp = (timestamps[frame_idx] - timestamps[0]).total_seconds()

        # Define string to display
        text = f"Frame: {frame_idx} | Time: {relative_timestamp:.2f}s"

        # Set text position & font
        position = (10, 30)  # Text position on the frame (top-left corner)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 0, 0)
        thickness = 2 

        # Display Text
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness)

        # Show frame at current timestamp
        cv2.imshow('Frame', frame)

        # Wait until next timestamp (unless final frame)
        if frame_idx > 0 and frame_idx < len(timestamps)-1:
            wait_time = (timestamps[frame_idx+1]-timestamps[frame_idx]).total_seconds()
            sleep_time = max(0, wait_time)  # Prevent negative sleep
            # time.sleep(sleep_time*1) # alter to find frames where blinks are

        # change value here to speed up finding blinks
        if frame_idx > 73:
            pass 
        
        # Increment frame counter
        frame_idx += 1
        
        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    end_time = datetime.now()
    print(f"Video playback lasted {(end_time-start_time).total_seconds()} seconds.")

    cap.release()
    cv2.destroyAllWindows()

# record_video(VIDEO_FILENAME,TIMESTAMP_FILENAME,VIDEO_DURATION)
play_video(VIDEO_FILENAME,TIMESTAMP_FILENAME)
