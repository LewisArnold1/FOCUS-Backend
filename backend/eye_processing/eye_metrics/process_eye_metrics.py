import os

from .face import FaceProcessor
from .blinks import BlinkProcessor
from .pupil import PupilProcessor
from .iris import IrisProcessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')
face_processor = FaceProcessor(PREDICTOR_PATH)
blink_processor = BlinkProcessor()
pupil_processor = PupilProcessor()
iris_processor = IrisProcessor()

def process_eye(frame):
    # Extract left and right eye landmarks
    no_faces, left_eye, right_eye = face_processor.process_face(frame)

    # If faces are not detected, return immediately
    if no_faces == 0:
        return no_faces, None, None, None  

    # Process blink detection
    blink_detected, avg_ear = blink_processor.process_blink(left_eye, right_eye)

    # Process pupil only if no blink is detected
    pupil = None if blink_detected else pupil_processor.process_pupil(frame, left_eye)

    return no_faces, blink_detected, avg_ear, pupil

