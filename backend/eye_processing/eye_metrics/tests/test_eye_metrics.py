import os
import sys
import cv2

# Adjust the path to ensure imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PARENT_DIR)

# Import the function to test
from pupil_tracking.process_eye_metrics import process_eye

def test_process_eye():
    # Initializse the video capture
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

        # Call the process_eye function
        total_blinks, ear, pupil, iris = process_eye(frame)
        print("Results:")
        print(f"Total Blinks: {total_blinks}, EAR: {ear}, Pupil: {pupil}, Iris: {iris}")

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting video feed...")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('here')
    test_process_eye()
