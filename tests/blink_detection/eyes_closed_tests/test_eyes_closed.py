import os
import cv2
import json
import csv
from datetime import datetime
import pandas as pd
import numpy as np

# change to first name & test number x of each saved video
VIDEO_FILENAME = "firstname_test_x.avi"
TIMESTAMP_FILENAME = "firstname_test_x_timestamps.txt"
IDEAL_FRAMES_FILENAME = "firstname_test_x_ideal.csv"
EAR_FILENAME = "firstname_test_x_ears.csv"
OUTPUT_FILENAME = "firstname_test_x_blinktype.csv"  # blinktype =  manual / auto / cnn

VIDEO_FILENAME = "soniya_test_3.avi"
TIMESTAMP_FILENAME = "soniya_test_3_timestamps.txt"
IDEAL_FRAMES_FILENAME = "soniya_test_3_ideal.csv"
EAR_FILENAME = "soniya_test_3_ears.csv"
OUTPUT_FILENAME = "soniya_test_3_auto.csv" # Manual: 25% | 50% | 75%, Auto: 0.4 | ... | 0.8

# Import the function to test
from process_eye_metrics import process_eye
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
    # elif len(timestamps) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
    #     print("Timestamps or frames missing.")
    #     return
    # else:
    #     print(f"Video has {len(timestamps)} frames/timestamps")

    # Process each frame
    frame_idx = 0
    ear_list = []
    while cap.isOpened():
        ret, frame = cap.read()

        # Stop at last frame
        if not ret or frame_idx >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            break

        _, ear, _ = process_eye(frame)
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
    eyes_closed_1 = []
    eyes_closed_2 = []
    eyes_closed_3 = []
    eyes_closed_4 = []
    eyes_closed_5 = []
    for i in range(len(ear_list)):
        n = 30 # Require n frames to create & apply threshold
        # calculate max from last n frames
        if len(ear_list[:i]) >= n:
            # calculate max from last n frames
            max  = sum(sorted(ear_list, reverse=True)[:n])/n
            # calculate different thresholds
            threshold_1 = max*0.4
            threshold_2 = max*0.5
            threshold_3 = max*0.6
            threshold_4 = max*0.7
            threshold_5 = max*0.8
            # Compare this frame with thresholds
            if ear_list[i] < threshold_1: # change as appropriate
                eyes_closed_1.append(1)
                eyes_closed_2.append(1)
                eyes_closed_3.append(1)
                eyes_closed_4.append(1)
                eyes_closed_5.append(1)
            elif ear_list[i] < threshold_2:
                eyes_closed_1.append(0)
                eyes_closed_2.append(1)
                eyes_closed_3.append(1)
                eyes_closed_4.append(1)
                eyes_closed_5.append(1)
            elif ear_list[i] < threshold_3:
                eyes_closed_1.append(0)
                eyes_closed_2.append(0)
                eyes_closed_3.append(1)
                eyes_closed_4.append(1)
                eyes_closed_5.append(1)
            elif ear_list[i] < threshold_4:
                eyes_closed_1.append(0)
                eyes_closed_2.append(0)
                eyes_closed_3.append(0)
                eyes_closed_4.append(1)
                eyes_closed_5.append(1)
            else:
                eyes_closed_1.append(0)
                eyes_closed_2.append(0)
                eyes_closed_3.append(0)
                eyes_closed_4.append(0)
                eyes_closed_5.append(1)

    # Save to CSV file
    data = zip(eyes_closed_1, eyes_closed_2, eyes_closed_3, eyes_closed_4, eyes_closed_5) # five columns
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

def calculate_metrics(ideal, eyes_closed_output):
    # Check arrays are same length
    if len(ideal) != len(eyes_closed_output):
        print("Output is wrong length")
        return
    
    # Calculate metrics using NumPy operations
    true_positives = np.sum((eyes_closed_output == 1) & (ideal == 1))
    false_positives = np.sum((eyes_closed_output == 1) & (ideal == 0))
    true_negatives = np.sum((eyes_closed_output == 0) & (ideal == 0))
    false_negatives = np.sum((eyes_closed_output == 0) & (ideal == 1))
            
    print(f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")
    precision = true_positives/(true_positives+false_positives)
    recall =  true_positives/(true_positives+false_negatives)
    F1_score = 2*precision*recall/(precision+recall)
    overall_accuracy = (true_positives+true_negatives)/len(ideal)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {F1_score:.3f}, Overall: {overall_accuracy:.3f}")
    return

def manual_metrics(ideal_filename, output_filename):
    # Set Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, "..","blink_test_files")
    ideal_path = os.path.join(tests_dir, ideal_filename)
    output_path = os.path.join(tests_dir, output_filename)

    # Retrieve ideal eyes_closed_list
    eyes_closed_ideal = pd.read_csv(ideal_path, header=None)
    eyes_closed_ideal = eyes_closed_ideal.iloc[:, 0].to_numpy()

    # Retrieve manual output & convert to 3 numpy arrays
    eyes_closed_output = pd.read_csv(output_path, header=None) 
    eyes_closed_25 = eyes_closed_output.iloc[:, 0].to_numpy()
    eyes_closed_50 = eyes_closed_output.iloc[:, 1].to_numpy()
    eyes_closed_75 = eyes_closed_output.iloc[:, 2].to_numpy()

    calculate_metrics(eyes_closed_ideal, eyes_closed_25)
    calculate_metrics(eyes_closed_ideal, eyes_closed_50)
    calculate_metrics(eyes_closed_ideal, eyes_closed_75)

    return

def calculate_metrics_segmented(ideal, eyes_closed_output):

    '''Below prints ideal & output for segment 3'''
    # for i in range(149,298):
    #     print(f"ideal: {ideal[i]}, output:{eyes_closed_output[i]}")

    '''Below calculates metrics for each segment'''
    # Check arrays are same length
    if len(ideal) != len(eyes_closed_output):
        print("Output is wrong length")
        return
    num_segments = 12 # 5s each
    segment_sizes = np.linspace(0, len(ideal), num_segments + 1, dtype=int)

    for i in range(num_segments):
        start, end = segment_sizes[i], segment_sizes[i + 1]
        segment_ideal = ideal[start:end]
        segment_output = eyes_closed_output[start:end]

        # Compute metrics for this segment
        true_positives = np.sum((segment_output == 1) & (segment_ideal == 1))
        false_positives = np.sum((segment_output == 1) & (segment_ideal == 0))
        true_negatives = np.sum((segment_output == 0) & (segment_ideal == 0))
        false_negatives = np.sum((segment_output == 0) & (segment_ideal == 1))
        precision = true_positives/(true_positives+false_positives)
        recall =  true_positives/(true_positives+false_negatives)
        # F1_score = 2*precision*recall/(precision+recall)
        # overall_accuracy = (true_positives+true_negatives)/len(segment_ideal)
        print(f"\nSegment no. {i+1}:")
        print(f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")
        # print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {F1_score:.3f}, Overall: {overall_accuracy:.3f}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
    
    return

def manual_metrics_segmented(ideal_filename, output_filename):
    # Set Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, "..","blink_test_files")
    ideal_path = os.path.join(tests_dir, ideal_filename)
    output_path = os.path.join(tests_dir, output_filename)

    # Retrieve ideal eyes_closed_list
    eyes_closed_ideal = pd.read_csv(ideal_path, header=None)
    eyes_closed_ideal = eyes_closed_ideal.iloc[:, 0].to_numpy()

    # Retrieve manual output & convert to 3 numpy arrays
    eyes_closed_output = pd.read_csv(output_path, header=None) 
    eyes_closed_25 = eyes_closed_output.iloc[:, 0].to_numpy()
    eyes_closed_50 = eyes_closed_output.iloc[:, 1].to_numpy()
    eyes_closed_75 = eyes_closed_output.iloc[:, 2].to_numpy()

    calculate_metrics_segmented(eyes_closed_ideal, eyes_closed_25)
    calculate_metrics_segmented(eyes_closed_ideal, eyes_closed_50)
    # calculate_metrics_segmented(eyes_closed_ideal, eyes_closed_75)

    return

def auto_metrics(ideal_filename, output_filename):
    # Set Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, "..","blink_test_files")
    ideal_path = os.path.join(tests_dir, ideal_filename)
    output_path = os.path.join(tests_dir, output_filename)

    # Retrieve ideal eyes_closed_list
    eyes_closed_ideal = pd.read_csv(ideal_path, header=None)
    eyes_closed_ideal = eyes_closed_ideal.iloc[:, 0].to_numpy()

    # Remove first n=30 frames
    eyes_closed_ideal = eyes_closed_ideal[30:]

    # Retrieve auto output & convert to __ numpy arrays
    eyes_closed_output = pd.read_csv(output_path, header=None) 
    eyes_closed_1 = eyes_closed_output.iloc[:, 0].to_numpy() # 0.4
    eyes_closed_2 = eyes_closed_output.iloc[:, 1].to_numpy() # 0.5
    eyes_closed_3 = eyes_closed_output.iloc[:, 2].to_numpy() # 0.6
    eyes_closed_4 = eyes_closed_output.iloc[:, 3].to_numpy() # 0.7
    eyes_closed_5 = eyes_closed_output.iloc[:, 4].to_numpy() # 0.8

    # Calculate metrics
    calculate_metrics(eyes_closed_ideal, eyes_closed_1)
    calculate_metrics(eyes_closed_ideal, eyes_closed_2)
    calculate_metrics(eyes_closed_ideal, eyes_closed_3)
    calculate_metrics(eyes_closed_ideal, eyes_closed_4)
    calculate_metrics(eyes_closed_ideal, eyes_closed_5)    

    return

def pop(ideal_filename):
    # Set Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, "..","blink_test_files")
    ideal_path = os.path.join(tests_dir, ideal_filename)

    # Retrieve ideal eyes_closed_list
    eyes_closed_ideal = pd.read_csv(ideal_path, header=None)

    # Remove the first row - change as required to clean data
    eyes_closed_ideal = eyes_closed_ideal.iloc[1:]

    # Save the modified data back to the same file
    eyes_closed_ideal.to_csv(ideal_path, header=False, index=False)

    print('done')




'''Calculate EAR at each frame, for all 9 videos'''
ear_list = calculate_ears(VIDEO_FILENAME,TIMESTAMP_FILENAME, EAR_FILENAME)

'''If outputs are 'no eye', please re-record video with better lighting!! - alternatively if for only few frames, data may be cleaned'''
# pop(IDEAL_FRAMES_FILENAME)

'''Run manual thresholding (including threshold sweep)'''
# test_manual(EAR_FILENAME, OUTPUT_FILENAME) # could still add smoothing filter?

'''Run auto thresholding (including threshold sweep)'''
# test_auto(EAR_FILENAME, OUTPUT_FILENAME)

'''Run CNN'''
# test_CNN(EAR_FILENAME, OUTPUT_FILENAME)

'''Test & Save Metrics for all'''
# manual_metrics(IDEAL_FRAMES_FILENAME, OUTPUT_FILENAME)
# auto_metrics(IDEAL_FRAMES_FILENAME, OUTPUT_FILENAME)

'''Segmented metrics - not currently included in report'''
# manual_metrics_segmented(IDEAL_FRAMES_FILENAME, OUTPUT_FILENAME)
