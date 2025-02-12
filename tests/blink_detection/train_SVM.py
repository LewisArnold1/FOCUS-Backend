import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

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

# Load EAR values and labels
def load_data(ears_filenames, labels_filenames):
    # Folder with test files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(script_dir, "blink_test_files")

    ear_values = [pd.read_csv(os.path.join(files_dir, file), header=None).values.flatten() for file in ears_filenames]
    labels = [pd.read_csv(os.path.join(files_dir, file), header=None).values.flatten() for file in labels_filenames]

    return ear_values, labels  # Returns lists of NumPy arrays

# Create feature matrix with a ±10 frame window
def create_feature_matrices(ear_values, labels, window_size=10):
    X, y = [], []
    for i in range(window_size, len(ear_values) - window_size):
        X.append(ear_values[i - window_size:i + window_size + 1])  # 21 values (±10 frames)
        y.append(labels[i])  # Label corresponds to center frame
    return np.array(X), np.array(y)

# Train and evaluate SVM model
def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Radial Basis Function kernel
    svm_model.fit(X_train, y_train)
    
    y_pred = svm_model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return svm_model, scaler

# Main function
def main(train_ears_filenames, train_labels_filenames, test_ears_filenames, test_labels_filenames):
    train_ear_values, train_labels = load_data(train_ears_filenames, train_labels_filenames)
    test_ear_values, test_labels = load_data(test_ears_filenames, test_labels_filenames)

    # train_feature_matrices, train_label_sets = create_feature_matrices(train_ear_values, train_labels)
    # test_feature_matrices, test_label_sets = create_feature_matrices(test_ear_values, test_labels)
    
    # models, scalers = train_svm(train_feature_matrices, train_label_sets)
    
    # for i, (model, scaler, X_test, y_test) in enumerate(zip(models, scalers, test_feature_matrices, test_label_sets)):
    #     X_test_scaled = scaler.transform(X_test)
    #     y_pred = model.predict(X_test_scaled)
        
    #     print(f"Test results for video {i+1}:")
    #     print("Accuracy:", accuracy_score(y_test, y_pred))
    #     print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # return models, scalers
    return

# Example usage
# model, scaler = main('ear_values.csv', 'labels.csv')

main(TRAIN_EARS_FILENAMES, TRAIN_LABELS_FILENAMES, TEST_EARS_FILENAMES, TEST_LABELS_FILENAMES)
