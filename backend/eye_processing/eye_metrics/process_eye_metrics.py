import cv2
import os
import traceback

from .face import FaceProcessor
from .iris import IrisProcessor
from .fixations_saccades import FixationSaccadeDetector

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')

face_processor = FaceProcessor()
iris_processor = IrisProcessor()
eye_movement_detector = FixationSaccadeDetector()

def process_eye(frame, timestamp_dt, blink_detected, **kwargs):
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Extract relevant keyword arguments
    draw_mesh = kwargs.get('draw_mesh', False)
    draw_contours = kwargs.get('draw_contours', False)
    show_axis = kwargs.get('show_axis', False)
    draw_eye = kwargs.get('draw_eye', False)
    filter = kwargs.get('filter', None)
    zoom = kwargs.get('zoom', False)

    face_detected, left_eye, right_eye, normalised_eye_speed, yaw, pitch, roll, diagnostic_frame, face_rect = face_processor.process_face(frame, draw_mesh=draw_mesh, draw_contours=draw_contours, show_axis=show_axis, draw_eye=draw_eye)
    focus = False

    if face_detected == 0 or (left_eye is None and right_eye is None):
        return face_detected, None, None, None, None, None, None, focus, None, None, "None", diagnostic_frame
    

    if (normalised_eye_speed > 0.25 or (abs(yaw) > 25 or abs(pitch) > 30)):
        return face_detected, normalised_eye_speed, yaw, pitch, roll, None, None, focus, None, None, "None", diagnostic_frame
    
    focus = True
    left_centre, right_centre = None, None
    left_iris_velocity, right_iris_velocity, movement_type = None, None, "None"

    if not blink_detected:
        try:
            left_centre = iris_processor.process_iris(frame, left_eye)
            right_centre = iris_processor.process_iris(frame, right_eye)
                    
            if filter == "eye":
                # Crop and draw pupil centers based on zoom mode
                cropped_frame = None
                if zoom:
                    # Crop around the eye region
                    cropped_frame = crop_around_eyes(frame, left_eye, right_eye)
                    draw_pupil_centers(cropped_frame, left_centre, right_centre)
                else:
                    # Crop around the face region
                    cropped_frame = crop_around_face(frame, face_rect)
                    draw_pupil_centers(cropped_frame, left_centre, right_centre)

                diagnostic_frame = cropped_frame

                # Process fixations and saccades
                left_iris_velocity, right_iris_velocity, movement_type = eye_movement_detector.process_eye_movements(
                    left_centre, right_centre, frame_width, frame_height, timestamp_dt
                )
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            return face_detected, normalised_eye_speed, yaw, pitch, roll, left_centre, right_centre, focus, None, None, "None", diagnostic_frame

    return face_detected, normalised_eye_speed, yaw, pitch, roll, left_centre, right_centre, focus, left_iris_velocity, right_iris_velocity, movement_type, diagnostic_frame

def crop_around_face(frame, face_rect):
    if face_rect is None:
        return frame  # Return original if no face detected

    x, y, w, h = face_rect
    cropped = frame[max(0, y): min(y + h, frame.shape[0]), max(0, x): min(x + w, frame.shape[1])]
    return cropped

def crop_around_eyes(frame, left_eye, right_eye, padding=20):
    if left_eye is None or right_eye is None:
        return frame  # Return original if eyes are not detected

    # Get min/max coordinates
    x_min = min(left_eye[:, 0].min(), right_eye[:, 0].min()) - padding
    y_min = min(left_eye[:, 1].min(), right_eye[:, 1].min()) - padding
    x_max = max(left_eye[:, 0].max(), right_eye[:, 0].max()) + padding
    y_max = max(left_eye[:, 1].max(), right_eye[:, 1].max()) + padding

    # Ensure coordinates are within frame bounds
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(frame.shape[1], x_max), min(frame.shape[0], y_max)

    cropped = frame[y_min:y_max, x_min:x_max]
    return cropped

def draw_pupil_centers(frame, left_centre, right_centre, radius=6, thickness=2):
    if left_centre is not None:
        cv2.circle(frame, left_centre, radius, (0, 255, 0), thickness, lineType=cv2.LINE_AA)
    if right_centre is not None:
        cv2.circle(frame, right_centre, radius, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

