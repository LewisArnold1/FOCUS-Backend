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
        
        self.prev_eye_positions = None 
        self.prev_rotation_matrix = None
        self.prev_time = None  

    def process_face(self, frame, draw=False, draw_mesh=True, draw_contours=True, draw_all=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return False, None, None, 0.0, None, None, None
        
        frame_height, frame_width, _ = frame.shape

        # Get face landmarks (main face)
        face_landmarks = results.multi_face_landmarks[0]

        # Compute face reference frame (axes)
        x_axis, y_axis, z_axis, yaw, pitch, roll = self.compute_face_axes(face_landmarks)

        # Convert main face coordinates to face frame
        face_rect, face_detected = self.extract_main_face(face_landmarks, frame_width, frame_height)
        left_eye, right_eye = self.extract_eye_regions(face_landmarks)

        # Compute normalised eye velocity in camera frame
        _, normalised_eye_speed = self.compute_velocity(left_eye, right_eye, x_axis, y_axis, z_axis)

        # Transform coordinates to pixel coordinates for plotting
        left_eye_pixels = self.convert_face_frame_to_pixels(left_eye, frame_width, frame_height)
        right_eye_pixels = self.convert_face_frame_to_pixels(right_eye, frame_width, frame_height)

        if not draw:
            return face_detected, left_eye_pixels, right_eye_pixels, normalised_eye_speed, yaw, pitch, roll

        if draw_all:
            self._draw_face_mesh(frame, face_landmarks, draw_mesh, draw_contours)
            self._draw_eye_annotations(frame, left_eye_pixels, right_eye_pixels, face_rect)
        elif draw_mesh or draw_contours:
            self._draw_face_mesh(frame, face_landmarks, draw_mesh, draw_contours)
        else:
            self._draw_eye_annotations(frame, left_eye_pixels, right_eye_pixels, face_rect)

        cv2.imshow("Face Landmarks", cv2.flip(frame, 1))

        return face_detected, left_eye_pixels, right_eye_pixels, normalised_eye_speed, yaw, pitch, roll
    
    def compute_face_axes(self, face_landmarks):
        # Extract key landmark positions (normalised coordinates)
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

        # Create rotation matrix (face to camera)
        R_face_to_camera = np.vstack([x_axis, y_axis, z_axis]).T  # 3x3 rotation matrix

        # Convert to Euler angles (yaw, pitch, roll)
        yaw = np.arctan2(R_face_to_camera[1, 0], R_face_to_camera[0, 0])  # Rotation around Z-axis
        pitch = np.arctan2(-R_face_to_camera[2, 0], np.sqrt(R_face_to_camera[2, 1]**2 + R_face_to_camera[2, 2]**2))  # Rotation around Y-axis
        roll = np.arctan2(R_face_to_camera[2, 1], R_face_to_camera[2, 2])  # Rotation around X-axis

        yaw, pitch, roll = np.degrees([yaw, pitch, roll]) 

        return x_axis, y_axis, z_axis, yaw, pitch, roll

    def extract_main_face(self, face_landmarks, frame_width, frame_height):
        if not face_landmarks:
            return None, False

        # Convert landmark points to pixel coordinates
        landmark_points = [(int(l.x * frame_width), int(l.y * frame_height)) for l in face_landmarks.landmark]
        
        # Compute bounding box
        x_min = min(p[0] for p in landmark_points)
        y_min = min(p[1] for p in landmark_points)
        x_max = max(p[0] for p in landmark_points)
        y_max = max(p[1] for p in landmark_points)

        return ((x_min, y_min, x_max - x_min, y_max - y_min), True)
    
    def extract_eye_regions(self, face_landmarks):
        LEFT_EYE_IDX = [33, 133, 160, 158, 153, 144]  # Left eye only
        RIGHT_EYE_IDX = [362, 263, 385, 387, 373, 380]  # Right eye only

        left_eye = np.array([
        (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z) for i in LEFT_EYE_IDX], dtype=np.float32)

        right_eye = np.array([
        (face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z) for i in RIGHT_EYE_IDX], dtype=np.float32)
        
        left_eye = self.sort_eye_landmarks(left_eye)
        right_eye = self.sort_eye_landmarks(right_eye)

        return left_eye, right_eye

    def sort_eye_landmarks(self, eye_points):
        centroid = np.mean(eye_points, axis=0)  # Get center of eye shape
        angles = np.arctan2(eye_points[:, 1] - centroid[1], eye_points[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)  # Sort by angle counterclockwise
        return eye_points[sorted_indices]
    
    def compute_velocity(self, left_eye, right_eye, x_axis, y_axis, z_axis):
        current_time = time.time()
        if self.prev_eye_positions is None or self.prev_time is None:
            self.prev_eye_positions = (left_eye, right_eye)
            self.prev_rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
            self.prev_time = current_time
            return np.array([0.0, 0.0, 0.0]), 0.0

        delta_t = current_time - self.prev_time
        if delta_t == 0:
            return np.array([0.0, 0.0, 0.0]), 0.0

        # Compute translational velocity for each landmark and average
        v_translational_left = np.mean((left_eye - self.prev_eye_positions[0]) / delta_t, axis=0)
        v_translational_right = np.mean((right_eye - self.prev_eye_positions[1]) / delta_t, axis=0)
        v_translational = (v_translational_left + v_translational_right) / 2

        # Compute angular velocity using finite differences on the rotation matrix
        R_current = np.vstack([x_axis, y_axis, z_axis]).T
        R_diff = R_current @ self.prev_rotation_matrix.T  # Relative rotation matrix
        angle = np.arccos((np.trace(R_diff) - 1) / 2.0)
        omega = (angle / delta_t) * np.array([R_diff[2, 1] - R_diff[1, 2],
                                              R_diff[0, 2] - R_diff[2, 0],
                                              R_diff[1, 0] - R_diff[0, 1]])

        # Compute rotational contribution to velocity
        r_eye_avg = (np.mean(left_eye, axis=0) + np.mean(right_eye, axis=0)) / 2
        v_rotational = np.cross(omega, r_eye_avg)

        # Compute total velocity
        v_total = v_translational + v_rotational

        # Store previous values
        self.prev_eye_positions = (left_eye, right_eye)
        self.prev_rotation_matrix = R_current
        self.prev_time = current_time
        
        # Compute Speed 
        speed = np.linalg.norm(v_total)
        
        return v_total, speed
    
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