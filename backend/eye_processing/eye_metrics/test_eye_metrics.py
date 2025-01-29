import os
import sys
import cv2

# Adjust the path to ensure imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PARENT_DIR)

# Import the function to test
from eye_processing.eye_metrics.process_eye_metrics import process_eye

def test_process_eye():
    # Initialise the video capture
    video = cv2.VideoCapture(0)
    print("here")
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        _, frame = video.read()
        if frame is None:
            print("Error: Could not read a video frame.")
            return

        (no_faces, normalised_face_speed, avg_ear, blink_detected, 
         left_pupil_centre, left_pupil_radius, right_pupil_centre, right_pupil_radius) = process_eye(frame)

        print(f"Faces: {no_faces}, Face Speed: {normalised_face_speed:.3f}, EAR: {avg_ear:.3f}, Blinking: {blink_detected}")
        print(f"Left Pupil: {left_pupil_centre}, Radius: {left_pupil_radius}")
        print(f"Right Pupil: {right_pupil_centre}, Radius: {right_pupil_radius}\n")


        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting video feed...")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('here')
    test_process_eye()
