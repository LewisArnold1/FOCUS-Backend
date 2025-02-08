import cv2
import dlib
from imutils import face_utils
import numpy as np
import time

class FaceProcessor:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.prev_center = None  
        self.prev_time = None  

    def extract_main_face(self, rects):
        # print(f"Number of faces detected: {len(rects)}")
        if not rects:
            return None, None
        return max(rects, key=lambda rect: rect.width() * rect.height()), len(rects)

    def extract_eye_regions(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        return left_eye, right_eye
    
    def compute_face_speed(self, face_rect, frame_height):
        current_time = time.time()  # Get current timestamp

        # Compute the center of the face bounding box
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        face_center = np.array([x + w // 2, y + h // 2])

        # If no previous data, store current position and return 0 speed
        if self.prev_center is None or self.prev_time is None:
            self.prev_center = face_center
            self.prev_time = current_time
            return 0.0

        # Compute displacement and velocity
        normalised_displacement = np.linalg.norm(face_center - self.prev_center) / frame_height  # Normalised Euclidean distance
        time_interval = current_time - self.prev_time

        normalised_speed = normalised_displacement / time_interval  # Pixels per second

        # Update previous values
        self.prev_center = face_center
        self.prev_time = current_time

        return normalised_speed

    def process_face(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_height = frame.shape[0]

        faces = self.detector(grey, 0)
        main_face, no_faces = self.extract_main_face(faces)
        if no_faces == 0 or main_face is None:
            return 0, None, None, 0.0
        
        shape = self.predictor(grey, main_face)
        shape = face_utils.shape_to_np(shape)

        normalised_face_speed = self.compute_face_speed(main_face, frame_height)

        left_eye, right_eye = self.extract_eye_regions(shape)
        
        return no_faces, left_eye, right_eye, normalised_face_speed