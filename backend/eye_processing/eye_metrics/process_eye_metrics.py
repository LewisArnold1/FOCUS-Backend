import cv2

from .face import FaceProcessor
from .blinks import BlinkProcessor
from .iris import IrisProcessor
from .fixations_saccades import FixationSaccadeDetector


face_processor = FaceProcessor()
blink_processor = BlinkProcessor()
iris_processor = IrisProcessor()
eye_movement_detector = FixationSaccadeDetector()


def process_eye(frame, timestamp_dt, draw_mesh=False, draw_contours=False, show_axis=False, draw_eye=False, verbose=0):
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    face_detected, left_eye, right_eye, normalised_eye_speed, yaw, pitch, roll, diagnostic_frame = face_processor.process_face(frame, draw_mesh=draw_mesh, draw_contour=draw_contours, show_axis=show_axis, draw_eye=draw_eye)
    focus = False

    if face_detected == 0:
        return face_detected, None, None, None, None, None, None, None, None, focus, diagnostic_frame

    blink_detected, avg_ear = blink_processor.process_blink(left_eye, right_eye)
    
    print(normalised_eye_speed)

    if (normalised_eye_speed > 0.25 or (abs(yaw) > 25 or abs(pitch) > 30)):
        return face_detected, normalised_eye_speed, yaw, pitch, roll, avg_ear, blink_detected, None, None, focus, diagnostic_frame
    
    focus = True
    left_centre, right_centre = None, None

    if not blink_detected:
        left_grey, left_colour, left_centre = iris_processor.process_iris(frame, left_eye)
        right_grey, right_colour, right_centre = iris_processor.process_iris(frame, right_eye)
        
        # Draw the raw pupil detection (before filtering)
        if left_centre is not None and right_centre is not None:
            cv2.circle(left_colour, left_centre, 5, (0, 0, 255), 1)
            cv2.circle(right_colour, right_centre, 5, (0, 0, 255), 1)

        # Display the images side by side (if verbose is set to 1)
        if verbose:
            iris_processor._display_images_in_grid(left_grey, left_colour, right_grey, right_colour)

    # Process fixations and saccades
    left_iris_velocity, right_iris_velocity, movement_type = eye_movement_detector.process_eye_movements(
        left_centre, right_centre, frame_width, frame_height, timestamp_dt
    ) 

    return face_detected, normalised_eye_speed, yaw, pitch, roll, avg_ear, blink_detected, left_centre, right_centre, focus, left_iris_velocity, right_iris_velocity, movement_type, diagnostic_frame

