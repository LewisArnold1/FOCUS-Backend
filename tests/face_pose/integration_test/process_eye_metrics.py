import cv2

from face import FaceProcessor
from blinks import BlinkProcessor
from iris import IrisProcessor

face_processor = FaceProcessor()
blink_processor = BlinkProcessor()
iris_processor = IrisProcessor()

def process_eye(frame, verbose=1):
    face_detected, left_eye, right_eye, normalised_face_speed, yaw, pitch, roll = face_processor.process_face(frame)
    focus = False

    if face_detected == 0:
        return face_detected, None, None, None, None, None, None, None, None, None, focus

    blink_detected, avg_ear = blink_processor.process_blink(left_eye, right_eye)

    if (normalised_face_speed > 0.3 or (abs(yaw) > 25 or abs(pitch) < 150)):
        return face_detected, normalised_face_speed, yaw, pitch, roll, avg_ear, blink_detected, None, None, focus
    
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
            iris_processor._display_images_in_grid(cv2.flip(left_grey, 1), cv2.flip(left_colour, 1), cv2.flip(right_grey, 1), cv2.flip(right_colour, 1))
        

    return face_detected, normalised_face_speed, yaw, pitch, roll, avg_ear, blink_detected, left_centre, right_centre, focus

    