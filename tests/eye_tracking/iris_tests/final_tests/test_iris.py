import cv2

# Import the function to test
from process_eye_metrics import process_eye

def test_process_eye():
    # Initialise the video capture
    video = cv2.VideoCapture(0)
    print("here")
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    counter = 0
    while True:
        _, frame = video.read()
        if frame is None:
            print("Error: Could not read a video frame.")
            return

        (no_faces, normalised_face_speed, avg_ear, blink_detected, 
         left_centre, right_centre) = process_eye(frame,1)

        print(f"Faces: {no_faces}, Face Speed: {normalised_face_speed}, EAR: {avg_ear}, Blinking: {blink_detected}")
        print(f"Left Iris: {left_centre}, Right Iris: {right_centre}\n")


        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting video feed...")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('here')
    test_process_eye()
