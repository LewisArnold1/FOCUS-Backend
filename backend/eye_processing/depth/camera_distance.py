import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

def process_frame(frame, detector):
    """
    Process a single frame, detect faces, and calculate depth.
    :param frame: Input image frame from the camera
    :param detector: FaceMeshDetector instance
    :return: Processed frame with depth annotations
    """
    img, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]

        # Calculate the distance between two points
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3  # Approximate width between the eyes in cm

        # Calculate depth
        f = 840  # Focal length (calibrated)
        d = (W * f) / w
        print(f"Depth: {d} cm")

        # Annotate the frame with depth information
        cvzone.putTextRect(img, f'Depth: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)
    return img


def main():
    """
    Main function to capture video and process frames.
    """q
    # Initialize video capture and detector
    cap = cv2.VideoCapture(1)
    detector = FaceMeshDetector(maxFaces=1)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image from webcam.")
            break

        # Process the current frame
        processed_frame = process_frame(frame, detector)

        # Display the processed frame
        cv2.imshow("Image", processed_frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to quit
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()