from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2
import numpy as np
import os
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

# Constants for blink detection
EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 4

# Initialize the blink detection counters
counter = 0
TOTAL = 0

# Initialize dlib's face detector and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
#detector = dlib.get_frontal_face_detector()

# Get the directory where count_blinks.py is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the shape predictor file
PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')

# Initialize dlib's face detector and the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor(PREDICTOR_PATH)

#from new threshold video
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
#idList = [23, 159]
ratioList = []
color = (255, 0, 255)
detector = FaceMeshDetector(maxFaces=1)

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
    img = cv2.resize(frame, (640,360))
    img, faces = detector.findFaceMesh(img, draw=False)

    color = (255,0, 255)
    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5,color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)
        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = int((lengthVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        if np.isnan(sum(ratioList) / len(ratioList)):
            ratioAvg = None
        else:
            ratioAvg = sum(ratioList) / len(ratioList)
            #if ratioAvg < 30:# and counter == 0:
            if ratioAvg < 28.5:
                TOTAL += 1
    else:
        ratioAvg = None
        #ear = ratioAvg
    '''
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    '''


    '''
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
        '''
    #return TOTAL, ear
    return TOTAL, ratioAvg