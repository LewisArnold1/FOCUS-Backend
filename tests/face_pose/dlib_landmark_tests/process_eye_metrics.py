import os
import cv2
import sys

from face import FaceProcessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
PREDICTOR_PATH = os.path.join(PARENT_DIR, 'shape_predictor_68_face_landmarks.dat')
print(PREDICTOR_PATH)
face_processor = FaceProcessor(PREDICTOR_PATH)

def process_eye(frame):
    no_faces, left_eye, right_eye, normalised_face_speed = face_processor.process_face(frame)

    if no_faces == 0:
        return

    