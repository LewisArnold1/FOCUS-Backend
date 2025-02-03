import os
import sys
import cv2
import time


# Adjust the path to ensure imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PARENT_DIR)


# change to first name & test number x of each saved video
VIDEO_FILENAME = "firstname_test_x.avi"
TIMESTAMP_FILENAME = "firstname_test_x_timestamps.txt"
# ^^ Make sure recorded .avi & .txt files are moved to eye_processing folder


# Import the function to test
from eye_processing.eye_metrics.process_eye_metrics import process_eye

'''

def test_process_eye(): # rewrite to use captured video
    # Initialize the video capture
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    
    eyes_closed = [0,0]
    total_blinks = 0
    while True:
        _, frame = video.read()
        if frame is None:
            print("Error: Could not read a video frame.")
            return

        # Call the process_eye function
        eye_closed, ear, pupil = process_eye(frame)
        eyes_closed.append(eye_closed)
        i = len(eyes_closed)-1
        print(i)
        print(eyes_closed[i])
        print(eyes_closed[i])
        if eyes_closed[i] == 1 and eyes_closed[i-1] == 0:
            total_blinks+=1
        # print("Results:")
        print(f"Total Blinks: {total_blinks}, Eye Closed: {eye_closed}, EAR: {ear}, Pupil: {pupil}")
        print(f"EAR: {ear}")
        #print(sum(eyes_closed))
        #time.sleep(1)    # Pause 5.5 seconds

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting video feed...")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_process_eye()

'''

def test_saved_video(timestamp_filename,video_filename):
    timestamp_path = timestamp_filename

    timestamps = []
    with open(timestamp_filename, 'r', encoding='utf-8') as f:  # Specify encoding explicitly
        for line in f:
            try:
                timestamp = float(line.strip())  # Convert string to float
                timestamps.append(timestamp)
            except ValueError:
                print(f"Error parsing line: {line.strip()}")
    return timestamps

    '''
    # Load timestamps
    timestamp_path = timestamp_filename
    
    with open(timestamp_path, "r", encoding="utf-8") as f:
        timestamps = [float(line.strip()) for line in f.readlines()]

    # Initialize the video capture
    video = cv2.VideoCapture(video_filename)
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    frame_idx = 0
    
    eyes_closed = []
    ears = []
    # total_blinks = 0
    while video.isOpened:
        ret, frame = video.read()
        if not ret or frame_idx>=len(timestamps):
            break

        # Call the process_eye function for each frame
        
        eye_closed, ear, pupil = process_eye(frame) # needs to be changed to send timestamp to process_eye
        eyes_closed.append(eye_closed)
        ears.append(ear)
        frame_idx += 1
        # i = len(eyes_closed)-1
        # print(i)
        # print(eyes_closed[i])
        # print(eyes_closed[i])
        # if eyes_closed[i] == 1 and eyes_closed[i-1] == 0:
        #     total_blinks+=1
        # # print("Results:")
        # print(f"Total Blinks: {total_blinks}, Eye Closed: {eye_closed}, EAR: {ear}, Pupil: {pupil}")
        # print(f"EAR: {ear}")
    '''
    # print(eyes_closed)

timestamps = test_saved_video(VIDEO_FILENAME,TIMESTAMP_FILENAME)

print(timestamps)
# print("Current directory:", os.getcwd())  # Print where Python is looking
# print("Files in directory:", os.listdir(os.getcwd()))  # Print available files
