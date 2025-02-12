import cv2
import mediapipe as mp
import numpy as np
import time

class FaceProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.prev_center = None  
        self.prev_time = None  

    def extract_main_face(self, face_landmarks, frame_width, frame_height):
        if not face_landmarks:
            return None, 0

        # Convert landmark points to pixel coordinates
        landmark_points = [(int(l.x * frame_width), int(l.y * frame_height)) for l in face_landmarks.landmark]
        
        # Compute bounding box
        x_min = min(p[0] for p in landmark_points)
        y_min = min(p[1] for p in landmark_points)
        x_max = max(p[0] for p in landmark_points)
        y_max = max(p[1] for p in landmark_points)

        return ((x_min, y_min, x_max - x_min, y_max - y_min), 1)
    
    def extract_eye_regions(self, face_landmarks):
        LEFT_EYE_IDX = [33, 133, 160, 158, 153, 144, 362, 385, 387, 263, 373, 380]  # Standard MediaPipe indices for left eye
        RIGHT_EYE_IDX = [362, 263, 385, 387, 373, 380, 33, 133, 160, 158, 153, 144]  # Standard MediaPipe indices for right eye

        left_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                         for i in LEFT_EYE_IDX], dtype=np.float32)

        right_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                          for i in RIGHT_EYE_IDX], dtype=np.float32)
        
        return left_eye, right_eye
    
    def compute_face_speed(self, face_rect, frame_height):
        current_time = time.time()  # Get current timestamp
        face_center = np.array([face_rect[0] + face_rect[2] // 2, face_rect[1] + face_rect[3] // 2])

        # If no previous data, store current position and return 0 speed
        if self.prev_center is None or self.prev_time is None:
            self.prev_center = face_center
            self.prev_time = current_time
            return 0.0

        # Compute displacement and velocity
        normalised_displacement = np.linalg.norm(face_center - self.prev_center) / frame_height
        time_interval = current_time - self.prev_time
        normalised_speed = normalised_displacement / time_interval  # Pixels per second

        # Update previous values
        self.prev_center = face_center
        self.prev_time = current_time

        return normalised_speed
    
    def process_face(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return 0, None, None, 0.0
        
        frame_height, frame_width, _ = frame.shape

        face_landmarks = results.multi_face_landmarks[0]  # Get the first (main) face
        face_rect, no_faces = self.extract_main_face(face_landmarks, frame_width, frame_height)
        left_eye, right_eye = self.extract_eye_regions(face_landmarks)
        normalised_face_speed = self.compute_face_speed(face_rect, frame_height)

        left_eye_pixels = np.array([(int(lm[0] * frame_width), int(lm[1] * frame_height)) for lm in left_eye])
        right_eye_pixels = np.array([(int(lm[0] * frame_width), int(lm[1] * frame_height)) for lm in right_eye])

        # 🔹 Draw face bounding box
        for (lx, ly) in left_eye_pixels:
            cv2.circle(frame, (lx, ly), 2, (0, 255, 255), -1)  # Yellow dots for left eye
        for (rx, ry) in right_eye_pixels:
            cv2.circle(frame, (rx, ry), 2, (0, 0, 255), -1)  # Red dots for right eye
            cv2.imshow("Face Landmarks", frame)

        return no_faces, left_eye, right_eye, normalised_face_speed
