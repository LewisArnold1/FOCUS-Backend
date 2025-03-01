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
    
    time_period = (max(timestamps) - min(timestamps)).total_seconds()
    if(time_period == 0):
        print('Time period is 0')
        return False
    fps = len(timestamps)/time_period

    # Check >= 12 FPS
    if fps < 12:
        print('FPS too low')
        return False
    
    # print(f"Time period recieved: {(max(timestamps) - min(timestamps)).total_seconds()}")

    # Identify centre EAR & timestamp
    centre_index = timestamps.index(middle_frame_timestamp)
    centre_time = timestamps[centre_index]
    centre_EAR = ear_values[centre_index]


    # Identify frames outside +- 0.35s
    start_time = centre_time - timedelta(seconds=0.35)
    end_time = centre_time + timedelta(seconds=0.35)
    before_indices = [i for i in range(len(timestamps)) if timestamps[i] >= start_time and timestamps[i] < centre_time]
    after_indices = [i for i in range(len(timestamps)) if timestamps[i] <= end_time and timestamps[i] > centre_time]
    
    # print('\n before changes')

    # print(f'before indices: {before_indices}')
    # print(f'after indices: {after_indices}')

    # # Require equal number of frames either side of centre frame
    # while len(before_indices) !=len(after_indices):
    #     print('not exactly 30 fps')
    #     # Remove any unequal frames
    #     if len(before_indices) > len(after_indices):
    #         before_indices = before_indices[1:]
    #     else:
    #         after_indices = after_indices[:-1]

    # Find frames before & after centre
    before_EARs = [ear_values[i] for i in before_indices]
    after_EARs = [ear_values[i] for i in after_indices]
    
    # print(f'\nno. frames before padding: {len(before_EARs)+len(after_EARs)+1}')
    # before = timestamps[before_indices[0]]
    # after = timestamps[after_indices[-1]]
    # print(f'time period {(after-before).total_seconds()}')
    # print(f'Intended time period {(end_time-start_time).total_seconds()}')

    # Pad or downsample so feature window has 10 frames either side of centre frame
    if len(before_indices) != 10:
        before_indices = np.linspace(0,len(before_EARs)-1, 10).astype(int)
        before_EARs = [ear_values[i] for i in before_indices]

    if len(after_indices) != 10:
        after_indices = np.linspace(0,len(after_EARs)-1, 10).astype(int)
        after_EARs = [ear_values[i] for i in after_indices]

    # print('after padding')
    # print(f'before indices: {before_indices}')
    # print(f'after indices: {after_indices}')

    feature_window = before_EARs + [centre_EAR] + after_EARs

    # print(f"SVM input: {len(feature_window)}")

    '''remove'''
    if len(feature_window) != 21:
        print('wrong feature window length')
        return False
    
    if any(EAR is None for EAR in feature_window):
        print('None values present')
        return False

    if np.any(np.isnan(np.array(feature_window))):
        print('NaN values present')
        return False
    
    feature_window = np.array(feature_window).reshape(1, -1)

    X_scaled = scaler.transform(feature_window)
    y_pred = svm_model.predict(X_scaled)

    if y_pred == True:
        print (f"                  Blink predicted")

    return y_pred
