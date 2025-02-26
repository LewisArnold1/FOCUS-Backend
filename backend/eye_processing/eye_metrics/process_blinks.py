import os
import numpy as np
import joblib

from .blinks import BlinkProcessor

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')
MODEL_PATH = os.path.join(CURRENT_DIR, "SVM_models", "svm_model_21.joblib")
SCALER_PATH = os.path.join(CURRENT_DIR, "SVM_models", "scaler_21.joblib")

# Load trained SVM model and scaler
try:
    svm_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("SVM model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading SVM model or scaler: {e}")
    svm_model, scaler = None, None  

blink_processor = BlinkProcessor(PREDICTOR_PATH)

def process_ears(frame):
    return blink_processor.process_ear(frame)

def process_blinks(ear_values):
    # if not svm_model or not scaler:
    #     print("Warning: SVM model or scaler not loaded. Defaulting to False.")
    #     return False  # Fallback to no blink detected if model is missing

    # # Extract the most recent window of EAR values
    # feature_window = np.array(ear_values).reshape(1, -1)

    # # Standardize the feature window using the pre-trained scaler
    # feature_window_scaled = scaler.transform(feature_window)

    # # Predict blink occurrence (1 = blink, 0 = no blink)
    # blink_prediction = svm_model.predict(feature_window_scaled)[0]

    # return bool(blink_prediction) 
    return False