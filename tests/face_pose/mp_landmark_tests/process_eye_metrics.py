import os
import cv2
import sys

from face import FaceProcessor

face_processor = FaceProcessor()

def process_eye(frame):
    no_faces, left_eye, right_eye, normalised_face_speed = face_processor.process_face(frame)

    if no_faces == 0:
        return

    