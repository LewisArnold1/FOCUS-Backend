import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
detector = FaceMeshDetector(maxFaces=1)

def process_blink(frame, user, session_id, video_id):
    threshold = 28.5 # change manually for now

    img = cv2.resize(frame, (640,360))
    img, faces = detector.findFaceMesh(img, draw=False)
    color = (255,0, 255)
    blink = 0       # Iniitialise as zero
    ratioAvg = None # Initialise as None
    ratio = None    # Initialise as None
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
        ratio = int((lengthVer / lenghtHor) * 100)
        
        # Smoothing Filter - average with prev 2 ratios
        from eye_processing.models import SimpleEyeMetrics

        this_video = SimpleEyeMetrics.objects.filter(user=user, session_id=session_id, video_id=video_id)
        video_ratios = list(this_video.values_list('eye_aspect_ratio', flat=True)) if this_video else []

        if len(video_ratios) >= 2:  # Ensure there are at least two previous frames
            ratio_list = [video_ratios[-2], video_ratios[-1], ratio]
            if any(r is None for r in ratio_list):  # Check for None values
                ratioAvg = None
            else:  # Calculate average
                ratioAvg = sum(ratio_list) / len(ratio_list)
                blink = 1 if ratioAvg < threshold else 0
    return blink, ratio, ratioAvg