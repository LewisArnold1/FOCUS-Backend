import os
import numpy as np
import joblib
from datetime import timedelta
from scipy.interpolate import interp1d

from .blinks import BlinkProcessor

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

def process_blinks(ear_values, timestamps, middle_frame_timestamp):
    if len(timestamps) != len(ear_values):
        print('Issue with process_blinks: timestamps & EAR unequal length')
        return False

    # Identify centre EAR & timestamp
    centre_index = timestamps.index(middle_frame_timestamp)
    centre_time = timestamps[centre_index]
    centre_EAR = ear_values[centre_index]


    # Identify frames outside +- 0.35s
    start_time = centre_time - timedelta(seconds=0.35)
    end_time = centre_time + timedelta(seconds=0.35)
    before_indices = [i for i in range(len(timestamps)) if timestamps[i] >= start_time and timestamps[i] < centre_time]
    after_indices = [i for i in range(len(timestamps)) if timestamps[i] <= end_time and timestamps[i] > centre_time]

    # Require equal number of frames either side of centre frame
    while len(before_indices) !=len(after_indices):
        # Remove any unequal frames
        if len(before_indices) > len(after_indices):
            before_indices = before_indices[1:]
        else:
            after_indices = after_indices[:-1]

    # Find frames before & after centre
    before_EARs = [ear_values[i] for i in before_indices]
    after_EARs = [ear_values[i] for i in after_indices]
    
    # For now check that code has worked
    if len(before_indices) != len(after_indices):
        print('zak is shit at coding') # remove this if code works lol
        return False
    
    # Check >= 11.4 FPS
    if len(before_indices) < 4:
        print('FPS too low')
        return False
    
    if len(before_indices) != 10:
    # > 30 FPS or > 30 FPS
        indices = np.linspace(0,len(before_EARs-1), 10).astype(int)
        before_EARs = [ear_values[i] for i in indices]
        after_EARs = [ear_values[i] for i in indices]

    feature_window = before_EARs + [centre_EAR] + after_EARs

    '''
    # Check if window has ~30fps: 25<fps<35 (0.65<t<0.84)
    window = (timestamps[-1] - timestamps[0]).total_seconds()

    valid_pairs = [(timestamps[i], ear_values[i]) for i in range(len(ear_values)) if ear_values[i] is not None]

    if len(valid_pairs) < 5:
        return False
    
    filtered_timestamps, filtered_ears = zip(*valid_pairs)
    selected_ears = np.array(filtered_ears)
    selected_times = np.array([(t - filtered_timestamps[0]).total_seconds() for t in filtered_timestamps])  # Convert to seconds

    if len(selected_ears) > 21:
        # Downsample: Select 21 evenly spaced frames
        downsample_indices = np.linspace(0, len(selected_ears) - 1, 21).astype(int)
        sampled_ears = selected_ears[downsample_indices]

    elif len(selected_ears) < 21:
        # Upsample: Use interpolation to create 21 points - MUST CHANGE!
        interp_func = interp1d(selected_times, selected_ears, kind='linear', fill_value="extrapolate")
        new_times = np.linspace(0, selected_times[-1], 21)  # Generate 21 equally spaced timestamps
        sampled_ears = interp_func(new_times)  # Interpolated EAR values

    else:
        sampled_ears = selected_ears  # Already 21 frames

    # Reshape for SVM prediction
    sampled_ears = np.array(sampled_ears).reshape(1, -1)  # Convert to 2D array

    # Scale and predict using the SVM model
    X_scaled = scaler.transform(sampled_ears)
    y_pred = svm_model.predict(X_scaled)

    '''
    X_scaled = scaler.transform(feature_window)
    y_pred = svm_model.predict(X_scaled)

        
    return y_pred
