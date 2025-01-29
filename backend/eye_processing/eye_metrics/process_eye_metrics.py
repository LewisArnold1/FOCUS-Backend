import os

from .face import FaceProcessor
from .blinks import BlinkProcessor
from .pupil import PupilProcessor
from .pupil import PupilTracker  

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')

face_processor = FaceProcessor(PREDICTOR_PATH)
blink_processor = BlinkProcessor()
pupil_processor = PupilProcessor()
pupil_tracker = PupilTracker()  

def process_eye(frame):
    no_faces, left_eye, right_eye, normalised_face_speed = face_processor.process_face(frame)

    if no_faces == 0:
        return no_faces, None, None, None, None, None, None, None

    blink_detected, avg_ear = blink_processor.process_blink(left_eye, right_eye)

    if normalised_face_speed > 0.2:
        return no_faces, normalised_face_speed, avg_ear, blink_detected, None, None, None, None

    left_pupil_centre, left_pupil_radius = None, None
    right_pupil_centre, right_pupil_radius = None, None

    if not blink_detected:
        left_pupil_centre, left_pupil_radius = pupil_processor.process_pupil(frame, left_eye)
        right_pupil_centre, right_pupil_radius = pupil_processor.process_pupil(frame, right_eye)

    left_pupil_centre, left_pupil_radius, right_pupil_centre, right_pupil_radius = pupil_tracker.update_pupil(
        left_pupil_centre, left_pupil_radius, right_pupil_centre, right_pupil_radius
    )

    return no_faces, normalised_face_speed, avg_ear, blink_detected, left_pupil_centre, left_pupil_radius, right_pupil_centre, right_pupil_radius
