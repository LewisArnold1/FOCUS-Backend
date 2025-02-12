import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set temporal window size
WINDOW_SIZE = 10 # 10 frames before & 10 frames after

# Files for training data
TRAIN_EARS_1 = "zak_test_1_ears.csv"
TRAIN_EARS_2 = "zak_test_2_ears.csv"
TRAIN_EARS_3 = "anaya_test_1_ears.csv"
TRAIN_EARS_4 = "anaya_test_2_ears.csv"
TRAIN_EARS_5 = "waasiq_test_1_ears.csv"
TRAIN_EARS_6 = "waasiq_test_2_ears.csv"
TRAIN_EARS_FILENAMES = np.array([TRAIN_EARS_1, TRAIN_EARS_2, TRAIN_EARS_3, TRAIN_EARS_4, TRAIN_EARS_5, TRAIN_EARS_6])

# Files for training labels
TRAIN_LABELS_1 = "zak_test_1_ideal.csv"
TRAIN_LABELS_2 = "zak_test_2_ideal.csv"
TRAIN_LABELS_3 = "anaya_test_1_ideal.csv"
TRAIN_LABELS_4 = "anaya_test_2_ideal.csv"
TRAIN_LABELS_5 = "waasiq_test_1_ideal.csv"
TRAIN_LABELS_6 = "waasiq_test_2_ideal.csv"
TRAIN_LABELS_FILENAMES = np.array([TRAIN_LABELS_1, TRAIN_LABELS_2, TRAIN_LABELS_3, TRAIN_LABELS_4, TRAIN_LABELS_5, TRAIN_LABELS_6])

# Files for testing data
TEST_EARS_1 = "zak_test_3_ears.csv"
TEST_EARS_2 = "anaya_test_3_ears.csv"
TEST_EARS_3 = "waasiq_test_3_ears.csv"
TEST_EARS_FILENAMES = np.array([TEST_EARS_1, TEST_EARS_2, TEST_EARS_3])

# Files for testing labels
TEST_LABELS_1 = "zak_test_3_ideal.csv"
TEST_LABELS_2 = "anaya_test_3_ideal.csv"
TEST_LABELS_3 = "waasiq_test_3_ideal.csv"
TEST_LABELS_FILENAMES = np.array([TEST_LABELS_1, TEST_LABELS_2, TEST_LABELS_3])


def load_data(ears_filenames, labels_filenames):
    # Folder with test files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(script_dir, "blink_test_files")

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

def train_svm(X, y):
    X_combined = np.vstack(X) # Stack predictors
    y_combined = np.hstack(y) # Stack labels
    
    # Scale Predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Fit linear SVM model
    svm_model = SVC(kernel='linear', C=1)
    svm_model.fit(X_scaled, y_combined)
    
    return svm_model, scaler

def test_svm(model, scaler, X_list, y_list):
    for i, (X, y) in enumerate(zip(X_list, y_list)):
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        print(f"Test results for video {i+1}:")
        print("Accuracy:", accuracy_score(y, y_pred))

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}")
        precision = tp/(tp+fp)
        recall =  tp/(tp+fn)
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}\n")
        # F1_score = 2*precision*recall/(precision+recall)
        # overall_accuracy = (true_positives+true_negatives)/len(segment_ideal)
        # print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {F1_score:.3f}, Overall: {overall_accuracy:.3f}")
        
        # print("Classification Report:\n", classification_report(y, y_pred))


def main(window_size, train_ears_filenames, train_labels_filenames, test_ears_filenames, test_labels_filenames):
    # Load data
    train_ear_values, train_labels = load_data(train_ears_filenames, train_labels_filenames)
    test_ear_values, test_labels = load_data(test_ears_filenames, test_labels_filenames)

    # Create temporal window frames X and corresponding labels y
    X_train, y_train = create_feature_matrices(train_ear_values, train_labels, window_size)
    X_test, y_test = create_feature_matrices(test_ear_values, test_labels, window_size)
    
    # Train model
    svm_model_1, scaler_1 = train_svm(X_train, y_train)

    # Test accuracy of model on training videos
    print('Training Accuracy:\n')
    test_svm(svm_model_1, scaler_1, X_train, y_train)
    
    # Test accuracy of model on test videos
    print('Test Accuracy:\n')
    test_svm(svm_model_1, scaler_1, X_test, y_test)

    # Save model
    return

main(WINDOW_SIZE, TRAIN_EARS_FILENAMES, TRAIN_LABELS_FILENAMES, TEST_EARS_FILENAMES, TEST_LABELS_FILENAMES)
