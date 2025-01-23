import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
detector = FaceMeshDetector(maxFaces=1)

class BlinkProcessor:
    def __init__(self):
        self.total = 0
        self.blink = 0


    def process_blink(self, frame, user, session_id, video_id):
        img = cv2.resize(frame, (640,360))
        img, faces = detector.findFaceMesh(img, draw=False)
        color = (255,0, 255)

        if faces:
            face = faces[0]

            # Left Eye
            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            left_length_Ver, _ = detector.findDistance(leftUp, leftDown)
            left_length_Hor, _ = detector.findDistance(leftLeft, leftRight)

            # Right Eye

            # Calculate eye-aspect ratio
            current_ear = int((left_length_Hor / left_length_Ver) * 100)
        else:
            current_ear = None

        from eye_processing.models import SimpleEyeMetrics
        # import EAR at each frame for current video
        this_video = SimpleEyeMetrics.objects.filter(user=user, session_id=session_id, video_id=video_id)
        video_ratios = list(this_video.values_list('eye_aspect_ratio', flat=True)) if this_video else []
        # Import if last frame was eyes closed or open        
        blinks = list(this_video.values_list('eyes_closed', flat=True)) if this_video else []
        prev_blink = blinks[-1] if blinks else None

        # remove None values
        clean_frames = [x for x in video_ratios if x is not None]
        # declare list for threshold calculation
        filteredList = []

        # calculate threshold from largest m values in previous n frames
        n = 120
        m = 10
        if len(video_ratios) == n:
            print(f'{n} frames reached')
        if len(clean_frames) >= n: # Ensure there are at least n previous frames with ear values
            for i in range(len(clean_frames)-n,len(clean_frames)):
                clean_list = [clean_frames[i-2], clean_frames[i-1], clean_frames[i]] # average EAR over prev 3 frames
                filteredList.append(sum(clean_list) / len(clean_list)) # append filtered average to list
            top_10_values = sorted(filteredList, reverse=True)[:m] # Largest 10 filtered values
            threshold = sum(top_10_values) / len(top_10_values) / 1.12 # find mean and multiply by factor for threshold
            #threshold = sum(top_10_values) / len(top_10_values) - 3.50 # subtract instead r
        else:
            threshold = None
        
        # determine if eyes are open
        if len(video_ratios) >= 2:  # Ensure there are at least two previous frames
            ratio_list = [video_ratios[-2], video_ratios[-1], current_ear]
            if any(r is None for r in ratio_list):  # Check for None values
                smoothed_ear = None
            else:  # Calculate average over prev 3 frames
                smoothed_ear = sum(ratio_list) / len(ratio_list)
                if threshold is not None:
                    if smoothed_ear < threshold:
                        self.blink = 1
                        if prev_blink == 0:
                            self.total +=1
                            print(f"total {self.total}, threshold {threshold}, ear {smoothed_ear}")
                    else:
                        self.blink = 0
            #print(f"smoothed_ear: {smoothed_ear}, threshold: {threshold}")
        return self.total, self.blink, current_ear
