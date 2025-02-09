import os

from face import FaceProcessor
from blinks import BlinkProcessor
from pupil import PupilProcessor
from iris import IrisProcessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../../'))
PREDICTOR_PATH = os.path.join(PARENT_DIR, 'shape_predictor_68_face_landmarks.dat')

face_processor = FaceProcessor(PREDICTOR_PATH)
blink_processor = BlinkProcessor()
pupil_processor = PupilProcessor()
iris_processor = IrisProcessor()

def process_eye(frame):
    # Extract left and right eye landmarks
    left_eye, right_eye = face_processor.process_face(frame)
    if left_eye is None or right_eye is None:
        print("No eye")
        return 0, None, None, None

    # Process blink detection
    total_blinks, ear = blink_processor.process_blink(left_eye, right_eye)

    # Process pupil coordinates
    pupil = pupil_processor.process_pupil(frame, left_eye)
    pupil = None

    # iris = iris_processor.process_iris(frame, left_eye)
    iris = None

    return total_blinks, ear, pupil, iris
