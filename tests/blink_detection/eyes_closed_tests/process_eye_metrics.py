import os
import sys
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

try:
    from .face import FaceProcessor
    from .blinks import BlinkProcessor
except ImportError:
    from face import FaceProcessor
    from blinks import BlinkProcessor

face_processor = FaceProcessor()
blink_processor = BlinkProcessor()

def process_eye(frame, draw_mesh=False, draw_contours=False, show_axis=False, draw_eye=False, verbose=0):
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    face_detected, left_eye, right_eye, normalised_eye_speed, yaw, pitch, roll, diagnostic_frame = face_processor.process_face(frame, draw_mesh=draw_mesh, draw_contours=draw_contours, show_axis=show_axis, draw_eye=draw_eye)
    focus = False

    if face_detected == 0 or (left_eye is None and right_eye is None):
        return face_detected, None

    avg_ear = blink_processor.process_blink(left_eye, right_eye)
    
    return face_detected, avg_ear


def process_eye_CNN(frame):
    # Extract left and right eye landmarks
    _, left_eye, right_eye, _ = face_processor.process_face(frame)
    if left_eye is None or right_eye is None:
        print("No eye")
        return 0, None

    closed = blink_processor.CNN(left_eye, right_eye)

    return closed