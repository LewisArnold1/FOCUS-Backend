import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

try:
    from .face import FaceProcessor
    from .blinks import BlinkProcessor
except ImportError:
    from face import FaceProcessor
    from blinks import BlinkProcessor

PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')

face_processor = FaceProcessor(PREDICTOR_PATH)
blink_processor = BlinkProcessor()

def process_eye(frame):
    # Extract left and right eye landmarks
    _, left_eye, right_eye = face_processor.process_face(frame)
    if left_eye is None or right_eye is None:
        print("No eye")
        return
    
    # Calculate EAR
    ear = blink_processor.eye_aspect_ratio(left_eye, right_eye)

    return ear

