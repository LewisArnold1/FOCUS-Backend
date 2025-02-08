import os
import cv2
import json
import csv
from datetime import datetime

# change to first name & test number x of each saved video
VIDEO_FILENAME = "firstname_test_x.avi"
TIMESTAMP_FILENAME = "firstname_test_x_timestamps.txt"
IDEAL_FRAMES_FILENAME = "firstname_test_x_ideal.csv"
EAR_FILENAME = "firstname_test_x_ears.csv"
OUTPUT_FILENAME = "firstname_test_x_blinktype.csv"
# blinktype_threshold = manual_25 /auto_x/cnn

VIDEO_FILENAME = "zak_test_2.avi"
TIMESTAMP_FILENAME = "zak_test_2_timestamps.txt"
IDEAL_FRAMES_FILENAME = "zak_test_2_ideal.csv"
EAR_FILENAME = "zak_test_2_ears.csv"
OUTPUT_FILENAME = "zak_test_2_manual.csv" # Manual: 25% | 50% | 75%

# Import the function to test
from process_eye_metrics import process_eye_manual
from process_eye_metrics import process_eye_auto
from process_eye_metrics import process_eye_CNN

def calculate_ears(video_filename,timestamp_filename,ear_filename):
    # Current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Tests folder
    tests_dir = os.path.join(script_dir, "..", "blink_test_files")

    # Full paths
    video_path = os.path.join(tests_dir, video_filename)   
    timestamp_path = os.path.join(tests_dir, timestamp_filename)
    ear_path = os.path.join(tests_dir, ear_filename)

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

    # Process each frame
    frame_idx = 0
    ear_list = []
    while cap.isOpened():
        ret, frame = cap.read()

        # Stop at last frame
        if not ret or frame_idx >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            break

        _, ear, _ = process_eye_manual(frame)
        if ear is None:
            print(frame_idx)
        else:
            ear_list.append(ear)
            
        # Increment frame counter        
        frame_idx += 1
    
    # Save output to CSV
    with open(ear_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for ear in ear_list:
            writer.writerow([ear])

    print('Done')

    return ear_list

def test_manual(ear_filename, output_filename):
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ear_path = os.path.join(script_dir, "..", "blink_test_files", ear_filename)
    output_path = os.path.join(script_dir, "..", "blink_test_files", output_filename)
    
    # Load ear_list
    ear_list = []
    with open(ear_path, 'r') as file:
        reader = csv.reader(file)
        for ear in reader:
            # Convert the value from string to float and add to the array
            ear_list.append(float(ear[0]))

    # Get average of largest and smallest 10 EAR values
    ear_max = sum(sorted(ear_list, reverse=True)[:10])/10
    ear_min = sum(sorted(ear_list)[:10])/10
    print(f"max = {ear_max:.3f}")
    print(f"min = {ear_min:.3f}")
    # Calculate each threshold
    threshold_25 = 0.25*(ear_max-ear_min)+ear_min
    threshold_50 = 0.5*(ear_max-ear_min)+ear_min
    threshold_75 = 0.75*(ear_max-ear_min)+ear_min
    # Display
    print(f"The 25% threshold is {threshold_25:.3f}")
    print(f"The 50% threshold is {threshold_50:.3f}")
    print(f"The 75% threshold is {threshold_75:.3f}")

    # Calculate eyes_closed for each treshold
    eyes_closed_25 = []
    eyes_closed_50 = []
    eyes_closed_75 = []
    for i in range(len(ear_list)):
        if ear_list[i] < threshold_25:
            eyes_closed_25.append(1)
            eyes_closed_50.append(1)
            eyes_closed_75.append(1)
        elif ear_list[i] < threshold_50:
            eyes_closed_25.append(0)
            eyes_closed_50.append(1)
            eyes_closed_75.append(1)
        elif ear_list[i] < threshold_75:
            eyes_closed_25.append(0)
            eyes_closed_50.append(0)
            eyes_closed_75.append(1)
        else:
            eyes_closed_25.append(0)
            eyes_closed_50.append(0)
            eyes_closed_75.append(0)

    # Save to CSV file
    data = zip(eyes_closed_25, eyes_closed_50, eyes_closed_75) # Three columns: 25% | 50% | 75%
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    return

def test_auto(ear_filename,output_filename):
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ear_path = os.path.join(script_dir, "..", "blink_test_files", ear_filename)
    output_path = os.path.join(script_dir, "..", "blink_test_files", output_filename)
    
    # Load ear_list
    ear_list = []
    with open(ear_path, 'r') as file:
        reader = csv.reader(file)
        for ear in reader:
            # Convert the value from string to float and add to the array
            ear_list.append(float(ear[0]))

    ''' Auto Threshold '''
    eyes_closed_1 = [] # change to name according to auto thresholds
    eyes_closed_2 = []
    eyes_closed_3 = []
    for i in range(len(ear_list)):
        # calculate max from last x frames
        if len(ear_list[:i]) > 50: # change 50
            # calculate max from last x frames
            max = 0
            # calculate different thresholds
            threshold_1 = 1
            threshold_2 = 2
            threshold_3 = 3
        # Compare this frame with thresholds
        if ear_list[i] < threshold_1: # change as appropriate
            eyes_closed_1.append(1)
            eyes_closed_2.append(1)
            eyes_closed_3.append(1)
        elif ear_list[i] < threshold_2:
            eyes_closed_1.append(0)
            eyes_closed_2.append(1)
            eyes_closed_3.append(1)
        elif ear_list[i] < threshold_3:
            eyes_closed_1.append(0)
            eyes_closed_2.append(0)
            eyes_closed_3.append(1)
        else:
            eyes_closed_1.append(0)
            eyes_closed_2.append(0)
            eyes_closed_3.append(0)

    # Save to CSV file
    data = zip(eyes_closed_1, eyes_closed_2, eyes_closed_3) # Three columns
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print('Done')

    return

# def test_CNN(video_filename,timestamp_filename,output_filename):
#     # Current directory
#     script_dir = os.path.dirname(os.path.abspath(__file__))

#     # Tests folder
#     tests_dir = os.path.join(script_dir, "..", "blink_test_files")

#     # Full paths
#     video_path = os.path.join(tests_dir, video_filename)   
#     timestamp_path = os.path.join(tests_dir, timestamp_filename)
#     output_path = os.path.join(tests_dir, output_filename)

#     # Load timestamps
#     if os.path.exists(timestamp_path):  # Check if the file exists
#         with open(timestamp_path, "r") as json_file:
#             timestamps_str = json.load(json_file)  # Load JSON data
#         timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') for ts in timestamps_str]  # Convert to datetime
#     else:
#         print("Timestamps file not found.")

#     # Load test video
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Cannot open video file.")
#         return
#     elif len(timestamps) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
#         print("Timestamps or frames missing.")
#         return
#     else:
#         print(f"Video has {len(timestamps)} frames/timestamps")
    
#     ''' CNN ''' # to be changed

#     # Process each frame
#     frame_idx = 0
#     eyes_closed_list = []
#     while cap.isOpened():
#         ret, frame = cap.read()

#         # Stop at last frame
#         if not ret or frame_idx >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
#             break
        
#         eye, _ = process_eye_CNN(frame)
#         eyes_closed_list.append(eye)

#         # Increment frame counter        
#         frame_idx += 1

#      # Save output to CSV

#     print('Done')

#     return eyes_closed_list # No EAR list for CNN

def metrics(eyes_closed_list, ideal_filename): # change to use CSV and do for all manual and save. Then auto too in a single CSV
    # Retrieve ideal eyes_closed_list
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, "blink_tests")
    ideal_path = os.path.join(tests_dir, ideal_filename)
    with open(ideal_path, "r") as file:
        ideal = [int(line.strip()) for line in file] # data in one column

    # Check arrays are same length
    if len(ideal) != len(eyes_closed_list):
        print("error with ideal")
        return
    
    # print(f"Ideal {ideal}")
    # print(f"Output {eyes_closed_list}")
    
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

    print(f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")
    precision = true_positives/(true_positives+false_positives)
    recall =  true_positives/(true_positives+false_negatives)
    F1_score = 2*precision*recall/(precision+recall)
    overall_accuracy = (true_positives+true_negatives)/len(ideal)

    return precision, recall, F1_score, overall_accuracy

'''Calculate EAR at each frame, for all 9 videos'''
ear_list = calculate_ears(VIDEO_FILENAME,TIMESTAMP_FILENAME, EAR_FILENAME)
'''If outputs are 'no eye', please re-record video with better lighting!!'''

'''Test manual thresholding (including threshold sweep)'''
test_manual(EAR_FILENAME, OUTPUT_FILENAME) # could still add smoothing filter?

'''Test auto thresholding'''
# test_auto(EAR_FILENAME, OUTPUT_FILENAME)

'''Test CNN'''
# test_CNN(EAR_FILENAME, OUTPUT_FILENAME)

'''Test & Save Metrics for all'''
# precision, recall, F1_score, overall_accuracy = metrics(OUTPUT_FILENAME, IDEAL_FRAMES_FILENAME)


# print(f"Precision: {precision:.3f},\nRecall: {recall:.3f},\n F1 Score: {F1_score:.3f},\nOverall Accuracy: {overall_accuracy:.3f}")
