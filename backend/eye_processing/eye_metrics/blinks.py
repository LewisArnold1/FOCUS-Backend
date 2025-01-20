import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
detector = FaceMeshDetector(maxFaces=1)

class BlinkProcessor:
    def __init__(self, eye_ar_thresh=28.5, eye_ar_consec_frames=4):
        self.eye_ar_thresh = eye_ar_thresh
        self.eye_ar_consec_frames = eye_ar_consec_frames
        self.counter = 0 # used for smoothing filter previously
        self.total = 0

    def process_blink(self, frame):
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
        ear = int((lengthVer / lenghtHor) * 100)

        # inclue smoothing filter later

        if ear <= self.eye_ar_thresh:
            self.total += 1

        return self.total, ear
