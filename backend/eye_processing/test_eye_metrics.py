import os
import cv2
import json
from datetime import datetime

# change to first name & test number x of each saved video
VIDEO_FILENAME = "firstname_test_x.avi"
TIMESTAMP_FILENAME = "firstname_test_x_timestamps.txt"
IDEAL_FRAMES_FILENAME = "firstname_test_x_ideal.csv"

# Import the function to test
from eye_metrics.process_eye_metrics import process_eye

'''
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
def test_saved_video(video_filename,timestamp_filename):
    # Current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Full paths
    video_path = os.path.join(script_dir, video_filename)   
    timestamp_path = os.path.join(script_dir, timestamp_filename)

    # Load timestamps
    if os.path.exists(timestamp_path):  # Check if the file exists
        with open(timestamp_path, "r") as json_file:
            timestamps_str = json.load(json_file)  # Load JSON data
        timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') for ts in timestamps_str]  # Convert to datetime
    else:
        print("Timestamps file not found.")

    # Load test video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    elif len(timestamps) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        print("Timestamps or frames missing.")
        return
    else:
        print(f"Video has {len(timestamps)} frames/timestamps")

    '''
    Uncomment Manual vs Auto as required
    '''

    ''' Manual Threshold '''
    # Process each frame
    frame_idx = 0
    eyes_closed_list = []
    ear_list = []
    while cap.isOpened():
        ret, frame = cap.read()

        # Stop at last frame
        if not ret or frame_idx >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            break

        eye, ear, _ = process_eye(frame)
        ear_list.append(ear)
        eyes_closed_list.append(eye)

        # Increment frame counter        
        frame_idx += 1
    return eyes_closed_list, ear_list

    ''' Auto Threshold '''
    # # Process each frame
    # frame_idx = 0
    # ear_list = []
    # eyes_closed_list = []
    # while cap.isOpened():
    #     ret, frame = cap.read()

    #     # Stop at last frame
    #     if not ret or frame_idx >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
    #         break
        
    #     eye, ear, _ = process_eye(frame, ear_list)
    #     eyes_closed_list.append(eye)
    #     ear_list.append(ear)

    #     # Increment frame counter        
    #     frame_idx += 1
    
    ''' CNN '''
    # To be added

    return eyes_closed_list

def metrics(eyes_closed_list, ideal_filename):
    # Retrieve ideal eyes_closed_list
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ideal_path = os.path.join(script_dir, ideal_filename)
    with open(ideal_path, "r") as file:
        ideal = [int(line.strip()) for line in file] # data in one column

    # Check arrays are same length
    if len(ideal) != len(eyes_closed_list):
        print("error with ideal")
        return
    
    print(f"Ideal {ideal}")
    print(f"Output {eyes_closed_list}")
    
    # Calculate metrics
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(len(ideal)):
        # Positives
        if eyes_closed_list[i] == 1:
            true_positives += 1 if ideal[i] == 1 else 0
            false_positives += 1 if ideal[i] == 0 else 0
        # Negatives
        else:
            true_negatives += 1 if ideal[i] == 0 else 0
            false_negatives += 1 if ideal[i] == 1 else 0

    precision = true_positives/(true_positives+false_positives)
    recall =  true_positives/(true_positives+false_negatives)
    F1_score = 2*precision*recall/(precision+recall)
    overall_accuracy = (true_positives+true_negatives)/len(ideal)

    return precision, recall, F1_score, overall_accuracy

'''Calculate EAR at each frame, for all videos'''
eyes_closed_list, ear_list = test_saved_video(VIDEO_FILENAME,TIMESTAMP_FILENAME)
'''If outputs are 'no eye', please re-record video with better lighting!!'''

'''If testing manual method use below to calculate the threshold to be used in blink.py''' # Comment all of this out during auto testing
# Get average of largest and smallest 10 EAR values
ear_max = sum(sorted(ear_list, reverse=True)[:10])/10
ear_min = sum(sorted(ear_list)[:10])/10
# Threshold sweep
threshold_25 = 0.25*(ear_max-ear_min)+ear_min
threshold_50 = 0.5*(ear_max-ear_min)+ear_min
threshold_75 = 0.75*(ear_max-ear_min)+ear_min
print(f"The 25% threshold is {threshold_25:.3f}")
print(f"The 50% threshold is {threshold_50:.3f}")
print(f"The 75% threshold is {threshold_75:.3f}")

'''Now sweep videos with each of the three thresholds'''
# precision, recall, F1_score, overall_accuracy = metrics(eyes_closed_list, IDEAL_FRAMES_FILENAME)


# May want to save eyes_closed_list to a csv for showing in appendix of paper?
# print(f"Precision: {precision},\nRecall: {recall},\n F1 Score: {F1_score},\nOverall Accuracy: {overall_accuracy}")
