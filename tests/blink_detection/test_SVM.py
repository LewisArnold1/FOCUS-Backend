import os
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

'''
Test metrics on both training and testing data
'''

# Files for training data
TRAIN_EARS_1 = "anaya_test_1_ears.csv"
TRAIN_EARS_2 = "anaya_test_2_ears.csv"
TRAIN_EARS_3 = "waasiq_test_1_ears.csv"
TRAIN_EARS_4 = "waasiq_test_2_ears.csv"
TRAIN_EARS_FILENAMES = np.array([TRAIN_EARS_1, TRAIN_EARS_2, TRAIN_EARS_3, TRAIN_EARS_4])

# Files for training timestamps
TRAIN_TIMESTAMPS_1 = "anaya_test_1_timestamps.txt"
TRAIN_TIMESTAMPS_2 = "anaya_test_2_timestamps.txt"
TRAIN_TIMESTAMPS_3 = "waasiq_test_1_timestamps.txt"
TRAIN_TIMESTAMPS_4 = "waasiq_test_2_timestamps.txt"
TRAIN_TIMESTAMPS_FILENAMES = np.array([TRAIN_TIMESTAMPS_1, TRAIN_TIMESTAMPS_2, TRAIN_TIMESTAMPS_3, TRAIN_TIMESTAMPS_4])

# Files for training labels
TRAIN_LABELS_1 = "anaya_test_1_ideal.csv"
TRAIN_LABELS_2 = "anaya_test_2_ideal.csv"
TRAIN_LABELS_3 = "waasiq_test_1_ideal.csv"
TRAIN_LABELS_4 = "waasiq_test_2_ideal.csv"
TRAIN_LABELS_FILENAMES = np.array([TRAIN_LABELS_1, TRAIN_LABELS_2, TRAIN_LABELS_3, TRAIN_LABELS_4])

# Files for testing data
TEST_EARS_1 = "zak_test_3_ears.csv"
TEST_EARS_2 = "anaya_test_3_ears.csv"
TEST_EARS_3 = "waasiq_test_3_ears.csv"
TEST_EARS_4 = "mahie_test_1_ears.csv"
TEST_EARS_5 = "mahie_test_2_ears.csv"
TEST_EARS_6 = "mahie_test_3_ears.csv"
TEST_EARS_7 = "soniya_test_1_ears.csv"
TEST_EARS_8 = "soniya_test_2_ears.csv"
TEST_EARS_9 = "soniya_test_3_ears.csv"
TEST_EARS_10 = "mahie_test_4_low_fps_ears.csv"
TEST_EARS_11 = "mahie_test_5_low_fps_ears.csv"
TEST_EARS_12 = "mahie_test_6_low_fps_ears.csv"
TEST_EARS_13 = "soniya_test_4_low_fps_ears.csv"
TEST_EARS_14 = "soniya_test_5_low_fps_ears.csv"
TEST_EARS_15 = "soniya_test_6_low_fps_ears.csv"
TEST_EARS_FILENAMES = np.array([TEST_EARS_1, TEST_EARS_2, TEST_EARS_3, TEST_EARS_4, TEST_EARS_5, TEST_EARS_6, TEST_EARS_7, TEST_EARS_8, TEST_EARS_9, TEST_EARS_10, TEST_EARS_11, TEST_EARS_12, TEST_EARS_13, TEST_EARS_14, TEST_EARS_15])

# Files for testing timestamps
TEST_TIMESTAMPS_1 = "zak_test_3_timestamps.txt"
TEST_TIMESTAMPS_2 = "anaya_test_3_timestamps.txt"
TEST_TIMESTAMPS_3 = "waasiq_test_3_timestamps.txt"
TEST_TIMESTAMPS_4 = "mahie_test_1_timestamps.txt"
TEST_TIMESTAMPS_5 = "mahie_test_2_timestamps.txt"
TEST_TIMESTAMPS_6 = "mahie_test_3_timestamps.txt"
TEST_TIMESTAMPS_7 = "soniya_test_1_timestamps.txt"
TEST_TIMESTAMPS_8 = "soniya_test_2_timestamps.txt"
TEST_TIMESTAMPS_9 = "soniya_test_3_timestamps.txt"
TEST_TIMESTAMPS_10 = "mahie_test_4_low_fps_timestamps.txt"
TEST_TIMESTAMPS_11 = "mahie_test_5_low_fps_timestamps.txt"
TEST_TIMESTAMPS_12 = "mahie_test_6_low_fps_timestamps.txt"
TEST_TIMESTAMPS_13 = "soniya_test_4_low_fps_timestamps.txt"
TEST_TIMESTAMPS_14 = "soniya_test_5_low_fps_timestamps.txt"
TEST_TIMESTAMPS_15 = "soniya_test_6_low_fps_timestamps.txt"
TEST_TIMESTAMPS_FILENAMES = np.array([TEST_TIMESTAMPS_1, TEST_TIMESTAMPS_2, TEST_TIMESTAMPS_3, TEST_TIMESTAMPS_4, TEST_TIMESTAMPS_5, TEST_TIMESTAMPS_6, TEST_TIMESTAMPS_7, TEST_TIMESTAMPS_8, TEST_TIMESTAMPS_9, TEST_TIMESTAMPS_10, TEST_TIMESTAMPS_11, TEST_TIMESTAMPS_12, TEST_TIMESTAMPS_13, TEST_TIMESTAMPS_14, TEST_TIMESTAMPS_15])

# Files for testing labels
TEST_LABELS_1 = "zak_test_3_ideal.csv"
TEST_LABELS_2 = "anaya_test_3_ideal.csv"
TEST_LABELS_3 = "waasiq_test_3_ideal.csv"
TEST_LABELS_4 = "mahie_test_1_ideal.csv"
TEST_LABELS_5 = "mahie_test_2_ideal.csv"
TEST_LABELS_6 = "mahie_test_3_ideal.csv"
TEST_LABELS_7 = "soniya_test_1_ideal.csv"
TEST_LABELS_8 = "soniya_test_2_ideal.csv"
TEST_LABELS_9 = "soniya_test_3_ideal.csv"
TEST_LABELS_10 = "mahie_test_4_low_fps_ideal.csv"
TEST_LABELS_11 = "mahie_test_5_low_fps_ideal.csv"
TEST_LABELS_12 = "mahie_test_6_low_fps_ideal.csv"
TEST_LABELS_13 = "soniya_test_4_low_fps_ideal.csv"
TEST_LABELS_14 = "soniya_test_5_low_fps_ideal.csv"
TEST_LABELS_15 = "soniya_test_6_low_fps_ideal.csv"
TEST_LABELS_FILENAMES = np.array([TEST_LABELS_1, TEST_LABELS_2, TEST_LABELS_3, TEST_LABELS_4, TEST_LABELS_5, TEST_LABELS_6, TEST_LABELS_7, TEST_LABELS_8, TEST_LABELS_9, TEST_LABELS_10, TEST_LABELS_11, TEST_LABELS_12, TEST_LABELS_13, TEST_LABELS_14, TEST_LABELS_15])

def load_data(ears_filenames, timestamps_filenames, labels_filenames):
    # Folder with test files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(script_dir, "blink_test_files")
    print(ears_filenames)

    # Load EAR values into array of arrays, containing all vid data - same for labels
    ear_values = [pd.read_csv(os.path.join(files_dir, file), header=None).values.flatten() for file in ears_filenames]
    labels = [pd.read_csv(os.path.join(files_dir, file), header=None).values.flatten() for file in labels_filenames]
    
    # Load timestamps and convert to datetime
    timestamps = []
    for file in timestamps_filenames:
        timestamp_path = os.path.join(files_dir, file)
        with open(timestamp_path, "r") as json_file:
            timestamps_str = json.load(json_file)  # Load JSON data (assuming it's JSON formatted)
        timestamps_file = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') for ts in timestamps_str]  # Convert to datetime
        timestamps.append(timestamps_file)

    return ear_values, timestamps, labels 

# Extract feature window and corresponding labels
def create_feature_matrices(ear_values_lists, timestamp_lists, labels_lists):
    X, y = [], []
    half_window = 10

    # Runs for each video so no window contains EAR values from two videos
    for ear_values, timestamps, labels in zip(ear_values_lists, timestamp_lists, labels_lists):
        X_video = []
        y_video = []

        # Create feature window for every frame in video (excluding first and last 10)
        for i in range(half_window, len(ear_values) - half_window):

            # Check if 21 frame window has ~30fps: 25<fps<35 (0.65<t<0.84)
            window = (timestamps[i+half_window] - timestamps[i-half_window]).total_seconds()
            if 0.6 < window and window < 0.84:
                # At ~30 FPS, use ±10 frames
                window_features = ear_values[i - half_window:i + half_window + 1] 
            else:
                # Get frames within 0.7s
                centre_time = timestamps[i]
                start_time = centre_time - timedelta(seconds=0.35)
                end_time = centre_time + timedelta(seconds=0.35)

                # Frames within +-0.35s
                before_indices = [j for j in range(i) if timestamps[j] >= start_time]
                after_indices = [j for j in range(i + 1, len(timestamps)) if timestamps[j] <= end_time]

                if window <= 0.6:
                # larger than 35 fps, downsample

                    # Check there are at least 10 frames before and 10 after
                    if len(before_indices) < half_window or len(after_indices) < half_window:
                        # need to append none here? - only happens at start or end of vid so can be ignored
                        continue

                    # Sample 10 frames within 0.35s before and 0.35s after
                    before_indices = np.linspace(before_indices[0], before_indices[-1], 10).astype(int)
                    after_indices = np.linspace(after_indices[0], after_indices[-1], 10).astype(int)
                    
                    # Combine indices and get corresponding ears
                    indices = np.concatenate([before_indices, [i], after_indices])
                    window_features = [ear_values[idx] for idx in indices]
                else:
                    # If less than 10 fps (actually 11.4)
                    # if len(before_indices) < 4 or len(after_indices) < 4:
                    #     # window_features = [None]*21
                    #     continue
                        # not decided on if i should append None or ignore
                # less than 25 fps
                    # extend EAR values to synthesise 21 frames from less than 21
                    before_indices = np.linspace(before_indices[0], before_indices[-1], 10).astype(int)
                    after_indices = np.linspace(after_indices[0], after_indices[-1], 10).astype(int)
                    indices = np.concatenate((before_indices, [i], after_indices))
                    window_features = window_features = [ear_values[idx] for idx in indices]

            # Append feature window and label for this frame
            X_video.append(window_features)
            y_video.append(labels[i])  # Label corresponds to centre frame in window

        # Append feature windows & labels for this video
        X.append(X_video)
        y.append(y_video)
    return X, y


def test_svm(model, scaler, X_list, y_list):
    for i, (X, y) in enumerate(zip(X_list, y_list)):
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        fps = len(y)/60
        print(f"Test results for video {i+1} ({fps:.1f} fps):")
        # print("Accuracy:", accuracy_score(y, y_pred))

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp)
        recall =  tp/(tp+fn)
        F1_score = 2*precision*recall/(precision+recall)
        print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {F1_score:.3f}, Overall: {accuracy_score(y, y_pred):.3f}\n")
        
        # print("Classification Report:\n", classification_report(y, y_pred))

def test_segments(model, scaler, X_list, y_list):
    num_segments = 12
    for i, (X_test, y_test) in enumerate(zip(X_list, y_list)):
        segment_size = len(X_test) // num_segments
        print(f"\nSegmented test results for video {i+1}:\n")
        
        for j in range(num_segments):
            start = j * segment_size
            end = (j + 1) * segment_size if j < num_segments - 1 else len(X_test)
            
            X_segment = X_test[start:end]
            y_segment = y_test[start:end]
            
            if len(X_segment) == 0:
                continue
            
            X_segment_scaled = scaler.transform(X_segment)
            y_pred = model.predict(X_segment_scaled)
            
            cm = confusion_matrix(y_segment, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            print(f"Segment {j+1}:")
            print(f"Accuracy: {accuracy_score(y_segment, y_pred)}")
            precision = tp/(tp+fp)
            recall =  tp/(tp+fn)
            print(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}. Precision: {precision:.3f}, Recall: {recall:.3f}\n")

            # if i == 1 and j == 3: # check specific segment of a video 
            #     test = np.array([y_segment,y_pred])
            #     print(test)


def main(test_ears_filenames, test_timestamp_filenames, test_labels_filenames):
    # Load data
    test_ear_values, test_timestamps, test_labels = load_data(test_ears_filenames, test_timestamp_filenames, test_labels_filenames)

    # Create temporal window frames X and corresponding labels y
    X_test, y_test = create_feature_matrices(test_ear_values, test_timestamps, test_labels)

    # Path to load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "SVM_models")
    model_path = os.path.join(models_dir, 'svm_model_21.joblib')
    scaler_path = os.path.join(models_dir, 'scaler_21.joblib')

    # Load model
    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Test accuracy of model on test videos
    test_svm(svm_model, scaler, X_test, y_test)

    # print('Accuracy with segmented test videos:')
    # test_segments(svm_model, scaler, X_test, y_test)
    
    return

# Test on training data - participants 1 & 2 video 3s
main(TRAIN_EARS_FILENAMES, TRAIN_TIMESTAMPS_FILENAMES, TRAIN_LABELS_FILENAMES)

# Test on testing data: participant 1 & 2 video 3s + participant 3 all videos
main(TEST_EARS_FILENAMES[1:6], TEST_TIMESTAMPS_FILENAMES[1:6], TEST_LABELS_FILENAMES[1:6])

# Test with participant 3 low fps (mahie 17,14,17)
main(TEST_EARS_FILENAMES[9:12], TEST_TIMESTAMPS_FILENAMES[9:12], TEST_LABELS_FILENAMES[9:12])

# Test with participant 4 (Soniya ~ 19 fps)
main(TEST_EARS_FILENAMES[6:9], TEST_TIMESTAMPS_FILENAMES[6:9], TEST_LABELS_FILENAMES[6:9])

# Test with participant 4 low fps (Soniya ~ 7 fps)
# main(TEST_EARS_FILENAMES[12:15], TEST_TIMESTAMPS_FILENAMES[12:15], TEST_LABELS_FILENAMES[12:15]) # - currently only one exists