import os

from .face import FaceProcessor
from .blinks import BlinkProcessor
from .pupil import PupilProcessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')
face_processor = FaceProcessor(PREDICTOR_PATH)
blink_processor = BlinkProcessor()
pupil_processor = PupilProcessor()

def process_eye(frame, prev_ears):

    # # Extract left and right eye landmarks
    # left_eye, right_eye = face_processor.process_face(frame)
    # if left_eye is None or right_eye is None:
    #     print("No eye")
    #     return 0, None, None

    # Process blink detection
    total_blinks, ear_list, ear = blink_processor.process_blink(frame, prev_ears)

    # Process pupil coordinates
    # pupil = pupil_processor.process_pupil(left_eye, right_eye)

    # return total_blinks, ears, pupil
    return total_blinks, ear_list, ear,  None