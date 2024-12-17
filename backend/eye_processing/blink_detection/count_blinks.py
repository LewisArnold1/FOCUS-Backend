import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
detector = FaceMeshDetector(maxFaces=1)

def process_blink(frame, user, session_id, video_id):

    img = cv2.resize(frame, (640,360))
    img, faces = detector.findFaceMesh(img, draw=False)
    color = (255,0, 255)
    blink = 0       # Iniitialise as zero
    ratioAvg = None # Initialise as None
    ratio = None    # Initialise as None
    threshold = None# Initialise as None
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

        clean_frames = [x for x in video_ratios if x is not None]
        filteredList = []
        # calculate threshold from previous 30 frames
        if len(clean_frames) >= 30: # Ensure there are at least 30 previous frames with ear values
            for i in range(len(clean_frames)-28,len(clean_frames)):
                clean_list = [clean_frames[i-2], clean_frames[i-1], clean_frames[i]] # average EAR over 3 frames
                filteredList.append(sum(clean_list) / len(clean_list)) # append filtered average to list
            top_10_values = sorted(filteredList, reverse=True)[:10] # Largest 10 filtered values
            threshold = sum(top_10_values) / len(top_10_values) / 1.12 # find mean and multiply by factor for threshold
            #threshold = sum(top_10_values) / len(top_10_values) - 3.50 # subtract instead
        
        if len(video_ratios) >= 2:  # Ensure there are at least two previous frames
            ratio_list = [video_ratios[-2], video_ratios[-1], ratio]
            if any(r is None for r in ratio_list):  # Check for None values
                ratioAvg = None
            else:  # Calculate average
                ratioAvg = sum(ratio_list) / len(ratio_list)
                if threshold is not None:
                    blink = 1 if ratioAvg < threshold else 0
                    print(f"Threshold: {threshold}") # for now print threshold
    return blink, ratio, ratioAvg