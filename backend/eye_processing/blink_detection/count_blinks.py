from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2
import numpy as np
import os
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

# Initialize the blink detection counters
counter = 0
TOTAL = 0

#from new threshold video
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
color = (255, 0, 255)
detector = FaceMeshDetector(maxFaces=1)

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

        ear = int((lengthVer / lenghtHor) * 100)
        ratioList.append(ear)
        if len(ratioList) > 3:
            ratioList.pop(0)
        if np.isnan(sum(ratioList) / len(ratioList)):
            earAvg = None
        else:
            earAvg = sum(ratioList) / len(ratioList)
            if earAvg < 28.5:
                TOTAL += 1
    else:
        earAvg = None
        
    return TOTAL, earAvg