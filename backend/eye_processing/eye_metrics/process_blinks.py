import os
import numpy as np
import joblib
from datetime import timedelta

from .blinks import BlinkProcessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')
MODEL_PATH = os.path.join(CURRENT_DIR, "SVM_models", "svm_model_23.joblib")
SCALER_PATH = os.path.join(CURRENT_DIR, "SVM_models", "scaler_23.joblib")

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
    return blink_processor.process_blink(frame)

def process_blinks(ear_values, timestamps):
    # Check if 21 frame window has ~30fps: 25<fps<35 (0.65<t<0.84)
    window = (timestamps[0] - timestamps[-1]).total_seconds()
    if 0.6 < window and window < 0.84:
        # At ~30 FPS, use ±10 frames
        window_features = ear_values 
    else:
        # Get frames within 0.7s
        centre_time = timestamps[10]
        start_time = centre_time - timedelta(seconds=0.35)
        end_time = centre_time + timedelta(seconds=0.35)

        # Frames within +-0.35s
        before_indices = [j for j in range(10) if timestamps[j] >= start_time]
        after_indices = [j for j in range(10 + 1, len(timestamps)) if timestamps[j] <= end_time]

        if window <= 0.6:
        # larger than 35 fps, downsample

            # Check there are at least 10 frames before and 10 after
            if len(before_indices) < 10 or len(after_indices) < 10:
                # need to append none here? - only happens at start or end of vid so can be ignored
                return 0

            # Sample 10 frames within 0.35s before and 0.35s after
            before_indices = np.linspace(before_indices[0], before_indices[-1], 10).astype(int)
            after_indices = np.linspace(after_indices[0], after_indices[-1], 10).astype(int)
            
            # Combine indices and get corresponding ears
            indices = np.concatenate([before_indices, [10], after_indices])
            window_features = [ear_values[idx] for idx in indices]
        else:
        # less than 25 fps
            # extend EAR values to synthesise 21 frames from less than 21
            before_indices = np.linspace(before_indices[0], before_indices[-1], 10).astype(int)
            after_indices = np.linspace(after_indices[0], after_indices[-1], 10).astype(int)
            indices = np.concatenate((before_indices, [10], after_indices))
            window_features = [ear_values[idx] for idx in indices]

        X_scaled = scaler.transform(window_features)
        y = svm_model.predict(X_scaled)

        return bool(y)


