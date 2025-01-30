import os
import cv2
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

from face import FaceProcessor
from blinks import BlinkProcessor
from pupil import PupilProcessor
from pupil import PupilTracker

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
        left_pupil_centre, left_pupil_radius, left_grey, left_binary = pupil_processor.process_pupil(frame, left_eye)
        right_pupil_centre, right_pupil_radius, right_grey, right_binary = pupil_processor.process_pupil(frame, right_eye)

        # Initialise default values for color images
        raw_pupil_image = cv2.cvtColor(left_grey, cv2.COLOR_GRAY2RGB)
        filtered_pupil_image = cv2.cvtColor(left_grey, cv2.COLOR_GRAY2RGB)

        # Draw the raw pupil detection (before filtering)
        if left_pupil_centre is not None and left_pupil_radius is not None:
            cv2.circle(raw_pupil_image, left_pupil_centre, left_pupil_radius, (0, 0, 255), 1)  # Red for raw

        # Update pupil using the Kalman filter
        left_pupil_centre, left_pupil_radius, right_pupil_centre, right_pupil_radius = pupil_tracker.update_pupil(left_pupil_centre, left_pupil_radius, right_pupil_centre, right_pupil_radius)

        # Draw the Kalman-filtered pupil detection
        if left_pupil_centre is not None and left_pupil_radius is not None:
            cv2.circle(filtered_pupil_image, left_pupil_centre, left_pupil_radius, (0, 255, 0), 1)  # Green for filtered

        # Display the images side by side
        pupil_processor._display_images_in_grid(left_binary, raw_pupil_image, right_binary, filtered_pupil_image)


    return no_faces, normalised_face_speed, avg_ear, blink_detected, left_pupil_centre, left_pupil_radius, right_pupil_centre, right_pupil_radius
