import os
import cv2
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

from .face import FaceProcessor
from .blinks import BlinkProcessor
from .iris import IrisProcessor

PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')

face_processor = FaceProcessor(PREDICTOR_PATH)
blink_processor = BlinkProcessor()
iris_processor = IrisProcessor()

def process_eye(frame, verbose=0):
    no_faces, left_eye, right_eye, normalised_face_speed = face_processor.process_face(frame)

    if no_faces == 0:
        return no_faces, None, None, None, None, None

    blink_detected, avg_ear = blink_processor.process_blink(left_eye, right_eye)

    if normalised_face_speed > 0.2:
        return no_faces, normalised_face_speed, avg_ear, blink_detected, None, None

    left_centre, right_centre = None, None

    if not blink_detected:
        left_grey, left_colour, left_centre = iris_processor.process_iris(frame, left_eye)
        right_grey, right_colour, right_centre = iris_processor.process_iris(frame, right_eye)

        # Draw the raw pupil detection (before filtering)
        if left_centre is not None and right_centre is not None:
            cv2.circle(left_colour, left_centre, 5, (0, 0, 255), 1)
            cv2.circle(right_colour, right_centre, 5, (0, 0, 255), 1)

        # Display the images side by side (if verbose is set to 1)
        if verbose:
            iris_processor._display_images_in_grid(left_grey, left_colour, right_grey, right_colour)


    return no_faces, normalised_face_speed, avg_ear, blink_detected, left_centre, right_centre
