import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
detector = FaceMeshDetector(maxFaces=1)

class BlinkProcessor:
    def __init__(self, eye_ar_thresh=27.5, eye_ar_consec_frames=4):
        self.eye_ar_thresh = eye_ar_thresh
        self.eye_ar_consec_frames = eye_ar_consec_frames # used for smoothing filter previously
        self.counter = 0 # used for smoothing filter previously
        self.total = 0

    def process_blink(self, frame, ears):
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
            # Calculate eye-aspect ratio
            current_ear = int((lengthVer / lenghtHor) * 100)
        else:
            current_ear = None

        # Calculate prev smoothed ear
        if -1 in ears: prev_ear = None
        else: prev_ear = np.mean(np.array(ears))

        # Smoothing filter
        if -1 in ears:                          # applies for first three frames
            for i in range(3):
                if ears[i] == -1:
                    ears[i] = current_ear
                    break
        if -1 in ears:                          # applies for first two frames
            smooth_ear=None
        else:
            ears[0] = ears[1]
            ears[1] = ears[2]
            ears[2] = current_ear
            smooth_ear = np.mean(np.array(ears))
            # Increment total if EAR below threshold (and prev EAR was not)
            if prev_ear is not None and prev_ear>self.eye_ar_thresh and smooth_ear <= self.eye_ar_thresh:
                self.total += 1

        return self.total, ears, smooth_ear
