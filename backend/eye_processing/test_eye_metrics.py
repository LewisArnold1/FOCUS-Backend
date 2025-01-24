import os
import sys
import cv2
import time
# Adjust the path to ensure imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PARENT_DIR)

# Import the function to test
from eye_processing.eye_metrics.process_eye_metrics import process_eye

def test_process_eye():
    # Initialize the video capture
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    eyes_closed = [0]
    total_blinks = 0
    while True:
        _, frame = video.read()
        if frame is None:
            print("Error: Could not read a video frame.")
            return

        # Call the process_eye function
        eye_closed, ear, pupil = process_eye(frame)
        eyes_closed.append(eye_closed)
        for i in range (1,len(eyes_closed)):
            if eyes_closed[i] == 1 and eyes_closed[i-1] == 0:
                total_blinks+=1
        # print("Results:")
        # print(f"Total Blinks: {total_blinks}, Eye Closed: {eye_closed}, EAR: {ear}, Pupil: {pupil}")
        print(f"EAR: {ear}")
        # print(sum(eyes_closed))
        time.sleep(1)    # Pause 5.5 seconds

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting video feed...")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_process_eye()
