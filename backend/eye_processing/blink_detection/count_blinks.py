from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2
import numpy as np
import os

# Constants for blink detection
EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 4

# Initialize the blink detection counters
COUNTER = 0
TOTAL = 0

# Initialize dlib's face detector and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()

# Get the directory where count_blinks.py is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the shape predictor file
PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')

# Initialize dlib's face detector and the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor(PREDICTOR_PATH)


# Get the indexes for the left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def process_blink(frame):
    global COUNTER, TOTAL

    # Initialize ear with a default value
    ear = None

    # Convert the frame to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    rects = detector(gray, 0)

    for rect in rects:
        # Get the facial landmarks and convert them to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Compute the EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Check if EAR is below the blink threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

    return TOTAL, ear