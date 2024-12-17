import cv2
import dlib
import numpy as np
from imutils import face_utils

class FaceProcessor:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def extract_main_face(self, rects):
        print(f"Number of faces detected: {len(rects)}")
        if not rects:
            return None
        return max(rects, key=lambda rect: rect.width() * rect.height())

    def extract_eye_regions(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        return left_eye, right_eye

    def process_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray, 0)
        main_face = self.extract_main_face(faces)
        if main_face is None:
            return None, None
        
        shape = self.predictor(gray, main_face)
        shape = face_utils.shape_to_np(shape)
        
        return self.extract_eye_regions(shape)