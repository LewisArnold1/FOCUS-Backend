import cv2
import mediapipe as mp
import numpy as np
import time

class FaceProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.default_specs = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        self.prev_center = None  
        self.prev_time = None  

    def process_face(self, frame, draw_mesh=True, draw_contours=True, draw_all=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return 0, None, None, 0.0
        
        frame_height, frame_width, _ = frame.shape

        # Get face landmarks (main face)
        face_landmarks = results.multi_face_landmarks[0]

        # Step 1: Compute face reference frame (axes)
        x_axis, y_axis, z_axis = self.compute_face_axes(face_landmarks)

        # Step 2: Convert main face coordinates to face frame
        face_rect, no_faces = self.extract_main_face(face_landmarks, frame_width, frame_height)
        left_eye, right_eye = self.extract_eye_regions(face_landmarks)

        # Step 3: Convert eye coordinates to the face reference frame
        left_eye_transformed = self.transform_to_face_frame(left_eye, face_landmarks, x_axis, y_axis, z_axis)
        right_eye_transformed = self.transform_to_face_frame(right_eye, face_landmarks, x_axis, y_axis, z_axis)

        # Step 4: Compute normalised face velocity in face frame
        # normalised_face_speed = self.compute_face_speed(face_rect, frame_height, x_axis, y_axis, z_axis)
        normalised_face_speed = 0.0

        # Step 5: Transform coordinates to pixel coordinates for plotting
        left_eye_pixels = self.convert_face_frame_to_pixels(left_eye, frame_width, frame_height)
        right_eye_pixels = self.convert_face_frame_to_pixels(right_eye, frame_width, frame_height)

        if draw_all:
            self._draw_face_mesh(frame, face_landmarks, draw_mesh, draw_contours)
            self._draw_eye_annotations(frame, left_eye_pixels, right_eye_pixels, face_rect)
        elif draw_mesh or draw_contours:
            self._draw_face_mesh(frame, face_landmarks, draw_mesh, draw_contours)
        else:
            self._draw_eye_annotations(frame, left_eye_pixels, right_eye_pixels, face_rect)

        cv2.imshow("Face Landmarks", cv2.flip(frame, 1))

        return no_faces, left_eye, right_eye, normalised_face_speed
    
    def compute_face_axes(self, face_landmarks):
        # Extract key landmark positions (normalized coordinates)
        nose_tip = np.array([face_landmarks.landmark[1].x,
                            face_landmarks.landmark[1].y,
                            face_landmarks.landmark[1].z])

        left_eye = np.array([face_landmarks.landmark[33].x, 
                            face_landmarks.landmark[33].y, 
                            face_landmarks.landmark[33].z])

        right_eye = np.array([face_landmarks.landmark[263].x, 
                            face_landmarks.landmark[263].y, 
                            face_landmarks.landmark[263].z])

        forehead = np.array([face_landmarks.landmark[10].x, 
                            face_landmarks.landmark[10].y, 
                            face_landmarks.landmark[10].z])

        # Compute x-axis: from left eye to right eye (left to right)
        x_axis = right_eye - left_eye
        x_axis /= np.linalg.norm(x_axis)  

        # Compute y-axis: from nose to forehead (down to up)
        y_axis = forehead - nose_tip
        y_axis /= np.linalg.norm(y_axis) 

        # Compute z-axis: orthogonal to x and y (right-hand rule)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)  

        return x_axis, y_axis, z_axis

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
        LEFT_EYE_IDX = [33, 133, 160, 158, 153, 144]  # Left eye only
        RIGHT_EYE_IDX = [362, 263, 385, 387, 373, 380]  # Right eye only

        left_eye = np.array([
        (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z) for i in LEFT_EYE_IDX], dtype=np.float32)

        right_eye = np.array([
        (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z) for i in RIGHT_EYE_IDX], dtype=np.float32)
        
        return left_eye, right_eye
    
    def transform_to_face_frame(self, points, face_landmarks, x_axis, y_axis, z_axis):
        nose_tip = np.array([face_landmarks.landmark[1].x,
                            face_landmarks.landmark[1].y,
                            face_landmarks.landmark[1].z])

        # Rotation matrix
        R = np.vstack([x_axis, y_axis, z_axis]).T

        # Transform all points into the face reference frame
        transformed_points = np.array([
            R.T @ (np.array([p[0], p[1], p[2]]) - nose_tip) for p in points
        ])

        return transformed_points

        
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
    
    def convert_face_frame_to_pixels(self, transformed_points, frame_width, frame_height):
        pixel_points = np.array([
            (int((p[0]) * frame_width), int((p[1]) * frame_height))
            for p in transformed_points
        ])
        return pixel_points

    def _draw_eye_annotations(self, frame, left_eye_pixels, right_eye_pixels, face_rect):
        x, y, w, h = face_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box around face
        
        for (lx, ly) in left_eye_pixels:
            cv2.circle(frame, (lx, ly), 2, (0, 255, 255), -1)  # Yellow dots for left eye
        for (rx, ry) in right_eye_pixels:
            cv2.circle(frame, (rx, ry), 2, (0, 0, 255), -1)  # Red dots for right eye

    def _draw_face_mesh(self, frame, face_landmarks, draw_mesh=True, draw_contours=True):
        if draw_mesh:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        if draw_contours:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.default_specs
            )
