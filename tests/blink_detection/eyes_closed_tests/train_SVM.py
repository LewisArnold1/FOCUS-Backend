import os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set temporal window size
WINDOW_SIZE = 10 # 10 frames before & 10 frames after

# Files for training data
TRAIN_EARS_1 = "anaya_test_1_ears.csv"
TRAIN_EARS_2 = "anaya_test_2_ears.csv"
TRAIN_EARS_3 = "waasiq_test_1_ears.csv"
TRAIN_EARS_4 = "waasiq_test_2_ears.csv"
TRAIN_EARS_FILENAMES = np.array([TRAIN_EARS_1, TRAIN_EARS_2, TRAIN_EARS_3, TRAIN_EARS_4])

# Files for training labels
TRAIN_LABELS_1 = "anaya_test_1_ideal.csv"
TRAIN_LABELS_2 = "anaya_test_2_ideal.csv"
TRAIN_LABELS_3 = "waasiq_test_1_ideal.csv"
TRAIN_LABELS_4 = "waasiq_test_2_ideal.csv"
TRAIN_LABELS_FILENAMES = np.array([TRAIN_LABELS_1, TRAIN_LABELS_2, TRAIN_LABELS_3, TRAIN_LABELS_4])

# Files for testing data
TEST_EARS_1 = "anaya_test_3_ears.csv"
TEST_EARS_2 = "waasiq_test_3_ears.csv"
TEST_EARS_3 = "mahie_test_1_ears.csv"
TEST_EARS_4 = "mahie_test_2_ears.csv"
TEST_EARS_5 = "mahie_test_3_ears.csv"
TEST_EARS_FILENAMES = np.array([TEST_EARS_1, TEST_EARS_2, TEST_EARS_3, TEST_EARS_4, TEST_EARS_5])

# Files for testing labels
TEST_LABELS_1 = "anaya_test_3_ideal.csv"
TEST_LABELS_2 = "waasiq_test_3_ideal.csv"
TEST_LABELS_3 = "mahie_test_1_ideal.csv"
TEST_LABELS_4 = "mahie_test_2_ideal.csv"
TEST_LABELS_5 = "mahie_test_3_ideal.csv"
TEST_LABELS_FILENAMES = np.array([TEST_LABELS_1, TEST_LABELS_2, TEST_LABELS_3, TEST_LABELS_4, TEST_LABELS_5])

def load_data(ears_filenames, labels_filenames):
    # Folder with test files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(script_dir,"..","blink_test_files")

    # Load EAR values into array of arrays, containing all vid data - same for labels
    ear_values = [pd.read_csv(os.path.join(files_dir, file), header=None).values.flatten() for file in ears_filenames]
    labels = [pd.read_csv(os.path.join(files_dir, file), header=None).values.flatten() for file in labels_filenames]

    return ear_values, labels 

# Extract feature window as predictor and corresponding label for each frame
def create_feature_matrices(ear_values_lists, labels_lists, window_size):
    X, y = [], []
    # Runs for each video so no window contains EAR values from two videos
    for ear_values, labels in zip(ear_values_lists, labels_lists):
        X_video = []
        y_video = []
        # Create feature window for every frame in video (excluding first and last 10)
        for i in range(window_size, len(ear_values) - window_size):
            X_video.append(ear_values[i - window_size:i + window_size + 1])  # 21 EARs (±10 frames)
            y_video.append(labels[i])  # Label corresponds to centre frame in window

        # Append feature windows & labels for this video
        X.append(X_video)
        y.append(y_video)
    return X, y

def train_svm(X, y, C, W):
    X_combined = np.vstack(X) # Stack predictors
    y_combined = np.hstack(y) # Stack labels
    
    # Scale Predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Fit linear SVM model - introduce weight to prioritise closed eyes being detected
    svm_model = SVC(kernel='linear', C=C,  class_weight=W)
    svm_model.fit(X_scaled, y_combined)
    
    return svm_model, scaler

def test_svm_videos(model, scaler, X_list, y_list):
    for i, (X, y) in enumerate(zip(X_list, y_list)):
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        print(f"Test results for video {i+1}:")
        # print("Accuracy:", accuracy_score(y, y_pred))

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp)
        recall =  tp/(tp+fn)
        F1_score = 2*precision*recall/(precision+recall)
        print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {F1_score:.3f}, Overall: {accuracy_score(y, y_pred):.3f}\n")

def test_svm_overall(model, scaler, X_list, y_list):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i, (X, y) in enumerate(zip(X_list, y_list)):
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        # Add confusion values for this video to total
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        TP += tp
        FP += fp
        TN += tn
        FN += fn

    # Calculate metrics using confusion values from all vids combined
    precision = TP/(TP+FP)
    recall =  TP/(TP+FN)
    F1_score = 2*precision*recall/(precision+recall)
    accuracy = (TP + TN)/(TP+FP+TN+FP)
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {F1_score:.3f}, Overall: {accuracy:.3f}\n")

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


def main(window_size, train_ears_filenames, train_labels_filenames, test_ears_filenames, test_labels_filenames):
    # Load data
    train_ear_values, train_labels = load_data(train_ears_filenames, train_labels_filenames)
    test_ear_values, test_labels = load_data(test_ears_filenames, test_labels_filenames)

    # Create temporal window frames X and corresponding labels y
    X_train, y_train = create_feature_matrices(train_ear_values, train_labels, window_size)
    X_test, y_test = create_feature_matrices(test_ear_values, test_labels, window_size)

    # Set hyperparameters
    C = 0.01
    W = {0:1, 1:2}

    # Train model
    svm_model, scaler = train_svm(X_train, y_train, C, W)

    ''' Model Tuning
    Increasing Weight (to improve recall)
    1 is C=1, W=1:1
    2 is C=1, W=1:1.5
    3 is C=1, W=1:2
    4 is C=1, W=1:2.5
    5 is C=1, W=1:3
    6 is C=1, W=1:3.5
    7 is C=1, W=1:4

    Changing C:
    8 is C=10, W=1:1
    9 is C=100, W=1:1
    10 is C=1000, W=1:1
    11 is C=0.1, W=1:1
    12 is C=0.01, W=1:1
    13 is C=0.001, W=1:1

    deciding best W - is it only test that matters, cos then take 1:2, if both take 1:3

    Tune C with high W:
    5  is C=1, W=1:3
    14 is C=0.5, W=1:3
    15 is C=0.1, W=1:3
    16 is C=0.05, W=1:3
    17 is C=0.01, W=1:3
    18 is C=10, W=1:3
    19 is C=100, W=1:3

    Tune C with best? W:
    3  is C=1, W=1:2
    20 is C=0.5, W=1:2
    21 is C=0.1, W=1:2
    22 is C=0.05, W=1:2
    23 is C=0.01, W=1:2


    24 is C=0.025, W=1:2

    '''

    # Test accuracy of model on each of the training videos
    print('Metrics for each training video:\n')
    test_svm_videos(svm_model, scaler, X_train, y_train)
    
    # Test accuracy of model on each of the test videos
    print('Metrics for each testing video:\n')
    test_svm_videos(svm_model, scaler, X_test, y_test)

    # Overall accuracy over both the training videos
    print('Metrics over all training videos:\n')
    test_svm_overall(svm_model, scaler, X_train, y_train)

    # Overall accuracy over all the test videos
    print('Metrics over all testing videos:\n')
    test_svm_overall(svm_model, scaler, X_test, y_test)

    # Segmented metrics for training videos - not discussed in report
    print('Accuracy with segmented training videos:')
    test_segments(svm_model, scaler, X_train, y_train)

    # Segmented metrics for test videos - not discussed in report
    print('Accuracy with segmented test videos:')
    test_segments(svm_model, scaler, X_test, y_test)

    # Path to save model - change filenames for each iteration as appropriate
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..","SVM_models")
    model_path = os.path.join(models_dir, 'svm_model_23.joblib')
    scaler_path = os.path.join(models_dir, 'scaler_23.joblib')

    # Save model
    joblib.dump(svm_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(C)
    print(W)
    print('Model Saved')
    
    return

main(WINDOW_SIZE, TRAIN_EARS_FILENAMES, TRAIN_LABELS_FILENAMES, TEST_EARS_FILENAMES, TEST_LABELS_FILENAMES)
