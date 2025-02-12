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

# Files for training labels
TRAIN_LABELS_1 = "zak_test_1_ideal.csv"
TRAIN_LABELS_2 = "zak_test_2_ideal.csv"
TRAIN_LABELS_3 = "anaya_test_1_ideal.csv"
TRAIN_LABELS_4 = "anaya_test_2_ideal.csv"
TRAIN_LABELS_5 = "waasiq_test_1_ideal.csv"
TRAIN_LABELS_6 = "waasiq_test_2_ideal.csv"

# Files for testing data
TEST_EARS_1 = "zak_test_3_ears.csv"
TEST_EARS_2 = "anaya_test_3_ears.csv"
TEST_EARS_3 = "waasiq_test_3_ears.csv"

# Files for testing labels
TEST_LABELS_1 = "zak_test_3_ideal.csv"
TEST_LABELS_2 = "anaya_test_3_ideal.csv"
TEST_LABELS_3 = "waasiq_test_3_ideal.csv"

# Load EAR values and labels
def load_data(ear_csv, labels_csv):
    ear_values = pd.read_csv(ear_csv, header=None).values.flatten()  # Assuming a single column of EAR values
    labels = pd.read_csv(labels_csv, header=None).values.flatten()  # Assuming a single column of labels
    return ear_values, labels

# Create feature matrix with a ±10 frame window
def create_feature_matrix(ear_values, labels, window_size=10):


    
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
def main(ear_csv, labels_csv):
    ear_values, labels = load_data(ear_csv, labels_csv)
    X, y = create_feature_matrix(ear_values, labels)
    model, scaler = train_svm(X, y)
    return model, scaler

# Example usage
# model, scaler = main('ear_values.csv', 'labels.csv')
