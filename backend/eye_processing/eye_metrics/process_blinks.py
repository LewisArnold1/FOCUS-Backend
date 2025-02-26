import os

from .blinks import BlinkProcessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')

blink_processor = BlinkProcessor(PREDICTOR_PATH)

def process_ears(frame):
    return blink_processor.process_blink(frame)

def process_blinks(ear_values):
    return False

