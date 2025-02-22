import os
import numpy as np
import pandas as pd
import csv
from datetime import datetime, timedelta

# Files for training video timestamps
TRAIN_TIMESTAMPS_1 = "anaya_test_1_timestamps.txt"
TRAIN_TIMESTAMPS_2 = "anaya_test_2_timestamps.txt"
TRAIN_TIMESTAMPS_3 = "waasiq_test_1_timestamps.txt"
TRAIN_TIMESTAMPS_4 = "waasiq_test_2_timestamps.txt"
TRAIN_TIMESTAMPS_FILENAMES = np.array([TRAIN_TIMESTAMPS_1, TRAIN_TIMESTAMPS_2, TRAIN_TIMESTAMPS_3, TRAIN_TIMESTAMPS_4])

# Files for training video eyes_closed
TRAIN_LABELS_1 = "anaya_test_1_ideal.csv"
TRAIN_LABELS_2 = "anaya_test_2_ideal.csv"
TRAIN_LABELS_3 = "waasiq_test_1_ideal.csv"
TRAIN_LABELS_4 = "waasiq_test_2_ideal.csv"
TRAIN_LABELS_FILENAMES = np.array([TRAIN_LABELS_1, TRAIN_LABELS_2, TRAIN_LABELS_3, TRAIN_LABELS_4])

# Files for training video outputs
TRAIN_OUTPUT_1 = "anaya_test_1_svm.csv"
TRAIN_OUTPUT_2 = "anaya_test_2_svm.csv"
TRAIN_OUTPUT_3 = "waasiq_test_1_svm.csv"
TRAIN_OUTPUT_4 = "waasiq_test_2_svm.csv"
TRAIN_OUTPUT_FILENAMES = np.array([TRAIN_OUTPUT_1, TRAIN_OUTPUT_2, TRAIN_OUTPUT_3, TRAIN_OUTPUT_4])

# Files for testing video timestamps
TEST_TIMESTAMPS_1 = "anaya_test_3_timestamps.txt"
TEST_TIMESTAMPS_2 = "waasiq_test_3_timestamps.txt"
TEST_TIMESTAMPS_3 = "mahie_test_1_timestamps.txt"
TEST_TIMESTAMPS_4 = "mahie_test_2_timestamps.txt"
TEST_TIMESTAMPS_5 = "mahie_test_3_timestamps.txt"
TEST_TIMESTAMPS_6 = "mahie_test_4_low_fps_timestamps.txt"
TEST_TIMESTAMPS_7 = "mahie_test_5_low_fps_timestamps.txt"
TEST_TIMESTAMPS_8 = "mahie_test_6_low_fps_timestamps.txt"
TEST_TIMESTAMPS_9 = "soniya_test_1_timestamps.txt"
TEST_TIMESTAMPS_10 = "soniya_test_2_timestamps.txt"
TEST_TIMESTAMPS_11 = "soniya_test_3_timestamps.txt"
TEST_TIMESTAMPS_12 = "soniya_test_4_low_fps_timestamps.txt"
TEST_TIMESTAMPS_13 = "soniya_test_5_low_fps_timestamps.txt"
TEST_TIMESTAMPS_14 = "soniya_test_6_low_fps_timestamps.txt"
TEST_TIMESTAMPS_FILENAMES = np.array([TEST_TIMESTAMPS_1, TEST_TIMESTAMPS_2, TEST_TIMESTAMPS_3, TEST_TIMESTAMPS_4, TEST_TIMESTAMPS_5, TEST_TIMESTAMPS_6, TEST_TIMESTAMPS_7, TEST_TIMESTAMPS_8, TEST_TIMESTAMPS_9, TEST_TIMESTAMPS_10, TEST_TIMESTAMPS_11, TEST_TIMESTAMPS_12, TEST_TIMESTAMPS_13, TEST_TIMESTAMPS_14])

# Files for testing video labels
TEST_LABELS_1 = "anaya_test_3_ideal.csv"
TEST_LABELS_2 = "waasiq_test_3_ideal.csv"
TEST_LABELS_3 = "mahie_test_1_ideal.csv"
TEST_LABELS_4 = "mahie_test_2_ideal.csv"
TEST_LABELS_5 = "mahie_test_3_ideal.csv"
TEST_LABELS_6 = "mahie_test_4_low_fps_ideal.csv"
TEST_LABELS_7 = "mahie_test_5_low_fps_ideal.csv"
TEST_LABELS_8 = "mahie_test_6_low_fps_ideal.csv"
TEST_LABELS_9 = "soniya_test_1_ideal.csv"
TEST_LABELS_10 = "soniya_test_2_ideal.csv"
TEST_LABELS_11 = "soniya_test_3_ideal.csv"
TEST_LABELS_12 = "soniya_test_4_low_fps_ideal.csv"
TEST_LABELS_13 = "soniya_test_5_low_fps_ideal.csv"
TEST_LABELS_14 = "soniya_test_6_low_fps_ideal.csv"
TEST_LABELS_FILENAMES = np.array([TEST_LABELS_1, TEST_LABELS_2, TEST_LABELS_3, TEST_LABELS_4, TEST_LABELS_5, TEST_LABELS_6, TEST_LABELS_7, TEST_LABELS_8, TEST_LABELS_9, TEST_LABELS_10, TEST_LABELS_11, TEST_LABELS_12, TEST_LABELS_13, TEST_LABELS_14])

# Files for test video outputs
TEST_OUTPUT_1 = "anaya_test_3_svm.csv"
TEST_OUTPUT_2 = "waasiq_test_3_svm.csv"
TEST_OUTPUT_3 = "mahie_test_1_svm.csv"
TEST_OUTPUT_4 = "mahie_test_2_svm.csv"
TEST_OUTPUT_5 = "mahie_test_3_svm.csv"
TEST_OUTPUT_6 = "mahie_test_4_low_fps_svm.csv"
TEST_OUTPUT_7 = "mahie_test_5_low_fps_svm.csv"
TEST_OUTPUT_8 = "mahie_test_6_low_fps_svm.csv"
TEST_OUTPUT_9 = "soniya_test_1_svm.csv"
TEST_OUTPUT_10 = "soniya_test_2_svm.csv"
TEST_OUTPUT_11 = "soniya_test_3_svm.csv"
TEST_OUTPUT_12 = "soniya_test_4_low_fps_svm.csv"
TEST_OUTPUT_13 = "soniya_test_5_low_fps_svm.csv"
TEST_OUTPUT_14 = "soniya_test_6_low_fps_svm.csv"
TEST_OUTPUT_FILENAMES = np.array([TEST_OUTPUT_1, TEST_OUTPUT_2, TEST_OUTPUT_3, TEST_OUTPUT_3, TEST_OUTPUT_5, TEST_OUTPUT_6, TEST_OUTPUT_7, TEST_OUTPUT_8, TEST_OUTPUT_9, TEST_OUTPUT_10, TEST_OUTPUT_11, TEST_OUTPUT_12, TEST_OUTPUT_13, TEST_OUTPUT_14])

def consecutive_frames(ideal_filenames, output_filenames):
    # Set Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, "..","blink_test_files")

    # For each video
    video_num = 1
    for (ideal_file,output_file) in zip(ideal_filenames,output_filenames):
        # Load ideal and predicted data
        closed_ideal = pd.read_csv(os.path.join(tests_dir, ideal_file), header=None).values.flatten()
        closed_pred = pd.read_csv(os.path.join(tests_dir, output_file), header=None).values.flatten()

        '''
        Ideal file only ever has 1s at blinks
        Blink is considered to be at the leading 1
        '''
        
        # Find blinks in ideal eyes-closed file
        blinks_ideal = np.zeros(len(closed_ideal))
        for i in range(1,len(closed_ideal)-1):
            if closed_ideal[i-1] == 0 and closed_ideal[i] == 1:
                blinks_ideal[i] = 1

        # Determine blinks from closed_pred with sweep of consecutive frames - do more!
        blinks_pred_cons_2 = np.zeros(len(closed_ideal))
        blinks_pred_cons_3 = np.zeros(len(closed_ideal))
        blinks_pred_cons_4 = np.zeros(len(closed_ideal))
        for i in range(4,len(closed_ideal)-1):
            # At least 2 cons frames
            if closed_ideal[i-2] == 0 and closed_ideal[i-1] == 1 and blinks_ideal[i] == 1:
                blinks_pred_cons_2[i] = 1

            # At least 3 cons frames
            if closed_ideal[i-3] == 0 and closed_ideal[i-2] == 1 and closed_ideal[i-1] == 1 and blinks_ideal[i] == 1:
                blinks_pred_cons_3[i] = 1

            # At least 4 cons frames
            if closed_ideal[i-4] == 0 and closed_ideal[i-3] == 1 and closed_ideal[i-2] == 1 and closed_ideal[i-1] == 1 and blinks_ideal[i] == 1:
                blinks_pred_cons_4[i] = 1
            
        # For every actual blink, check if a blink was predicted within +-0.25s (+-7 frames)
        correctly_detected_2 = 0
        correctly_detected_3 = 0
        correctly_detected_4 = 0
        for i in range(len(blinks_ideal)):
            if blinks_ideal[i]==1:
                # Check if a blink has been detected for each method
                if any(blinks_pred_cons_2[max(0,i-7):min(len(blinks_ideal),i+8)]):
                    correctly_detected_2 +- 1
                if any(blinks_pred_cons_3[max(0,i-7):min(len(blinks_ideal),i+8)]):
                    correctly_detected_3 +- 1
                if any(blinks_pred_cons_4[max(0,i-7):min(len(blinks_ideal),i+8)]):
                    correctly_detected_4 +- 1
        
        # For every blink detected, check if there is suppsoed to be a detected blink within +-0.25s (+-7 frames)
        excessively_detected_2 = 0
        excessively_detected_3 = 0
        excessively_detected_4 = 0
        for i in range(len(blinks_ideal)):
            if blinks_pred_cons_2[i]==1 and sum(blinks_ideal[max(0,i-7):min(len(blinks_ideal),i+8)])==0:
                    excessively_detected_2 += 1
            if blinks_pred_cons_3[i]==1 and sum(blinks_ideal[max(0,i-7):min(len(blinks_ideal),i+8)])==0:
                    excessively_detected_3 += 1
            if blinks_pred_cons_4[i]==1 and sum(blinks_ideal[max(0,i-7):min(len(blinks_ideal),i+8)])==0:
                    excessively_detected_4 += 1
        
        # Output results
        print(f"Video number: {video_num}")
        print(f"Correctly detected: {correctly_detected_2} (with 2 cons)")
        print(f"Correctly detected: {correctly_detected_3} (with 3 cons)")
        print(f"Correctly detected: {correctly_detected_4} (with 4 cons)")
        print(f"Out of {sum(blinks_ideal)} actual blinks")
        print(f"Incorrectly detected: {excessively_detected_2} (with 2 cons)")
        print(f"Incorrectly detected: {excessively_detected_3} (with 3 cons)")
        print(f"Incorrectly detected: {excessively_detected_4} (with 4 cons)")

        video_num += 1
        

def consecutive_time(ideal_filenames, output_filenames, timestamp_filenames):
    # Set Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, "..","blink_test_files")

    # For each video
    for (ideal_file,output_file) in zip(ideal_filenames,output_filenames):
        # Load ideal and predicted data
        closed_ideal = pd.read_csv(os.path.join(tests_dir, ideal_file), header=None).values.flatten()
        closed_pred = pd.read_csv(os.path.join(tests_dir, output_file), header=None).values.flatten()

        '''
        Ideal file only ever has 1s at blinks
        Blink is considered to be at the leading 1
        '''
        
        # Find blinks in ideal eyes-closed file
        blinks_ideal = np.zeros(len(closed_ideal))
        for i in range(1,len(closed_ideal)-1):
            if closed_ideal[i-1] == 0 and blinks_ideal[i] == 1:
                blinks_ideal[i] = 1
        print(blinks_ideal)

        # Determine blinks from closed_pred with sweep on consecutive time


def main(ideal_filenames, output_filenames, timestamp_filenames):
    consecutive_frames(ideal_filenames, output_filenames)
    # consecutive_time(ideal_filenames, output_filenames, timestamp_filenames)

main(TRAIN_LABELS_FILENAMES, TRAIN_OUTPUT_FILENAMES, TRAIN_TIMESTAMPS_FILENAMES)     
