import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

try:
    from .face import FaceProcessor
    from .blinks import BlinkProcessor
    from .iris import IrisProcessor
except ImportError:
    from face import FaceProcessor
    from blinks import BlinkProcessor
    from iris import IrisProcessor

PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')

face_processor = FaceProcessor(PREDICTOR_PATH)
blink_processor = BlinkProcessor()
iris_processor = IrisProcessor()

def process_eye(frame):

    # Extract left and right eye landmarks
    _, left_eye, right_eye, _ = face_processor.process_face(frame)
    if left_eye is None or right_eye is None:
        print("No eye")
        return 0, None, None
    
    # Process pupil coordinates
    # pupil = pupil_processor.process_pupil(frame, left_eye)

    pupil = None

def process_eye_CNN(frame):
    # Extract left and right eye landmarks
    _, left_eye, right_eye, _ = face_processor.process_face(frame)
    if left_eye is None or right_eye is None:
        print("No eye")
        return 0, None, None
    
    # Process pupil coordinates - ignored for blink test
    pupil = None

    closed = blink_processor.CNN(left_eye, right_eye)

    return closed