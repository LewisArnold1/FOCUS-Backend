import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

def calculate_depth(frame):
    """
    Takes a frame, detects a face, and calculates the depth.
    :param frame: Input image frame
    :return: Depth in cm, or None if no face is detected
    """
    detector = FaceMeshDetector(maxFaces=1)
    img, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]

        # Calculate the distance between two points (eyes)
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3  # Approximate width between the eyes in cm

        # Calculate depth
        f = 840  # Focal length (calibrated)
        d = (W * f) / w
        return depth  # Return depth in cm

    return None  # Return None if no face is detected
