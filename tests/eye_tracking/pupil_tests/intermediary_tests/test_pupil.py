import cv2

# Import the function to test
from process_eye_metrics import process_eye

def test_process_eye():
    # Initialise the video capture
    video = cv2.VideoCapture(0)
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
    test_process_eye()
