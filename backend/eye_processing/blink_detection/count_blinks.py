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
#idList = [23, 159]
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
#def process_blink(frame, user, session_id, video_id):
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
            if ratioAvg < 28.5:
                blink = 1
            else:
                blink = 0
        '''
        # List of blinks in each frame for this video
        this_video = SimpleEyeMetrics.objects.filter(user=self.user,session_id=self.session_id,video_id=self.video_id)            
        # if prev frames exist
        if this_video:
            video_blinks = this_video.objects.values_list('blink_count')

            # if prev frame was also a blink, set blink = 0
            total_blinks = sum(list(video_blinks))+blink
        else:
            total_blinks = 0
        '''
        
    else:
        ratioAvg = None
        blink = 0
    #return TOTAL, ear
    return blink, ratioAvg