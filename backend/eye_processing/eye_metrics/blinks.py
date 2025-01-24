from scipy.spatial import distance as dist

class BlinkProcessor:
    def __init__(self, eye_ar_thresh=0.16, eye_ar_consec_frames=2):
        self.eye_ar_thresh = eye_ar_thresh
        self.eye_ar_consec_frames = eye_ar_consec_frames
        self.counter = 0
        # self.total = 0

    @staticmethod
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def process_blink(self, left_eye, right_eye, ear_list): # ear_list to be removed after testing

        # Note: ear_list is not required as an input in the website as it is read from database
        # But for now it must be passed from test_eye_metrics.py

        # Calculate threshold
        clean_frames = [x for x in ear_list if x is not None]
        filteredList = []
        # Calculate threshold from largest m values in previous n frames
        n = 30
        m = 10
        if len(clean_frames) >= n: # Ensure there are at least 30 previous frames with ear values
            for i in range(len(clean_frames)-(n-2),len(clean_frames)):
                clean_list = [clean_frames[i-2], clean_frames[i-1], clean_frames[i]] # average EAR over 3 frames
                filteredList.append(sum(clean_list) / len(clean_list)) # append filtered average to list
            top_m_values = sorted(filteredList, reverse=True)[:m] # Largest m filtered values
            threshold = sum(top_m_values) / len(top_m_values) / 1.3 # find mean and multiply by factor for threshold
        else:
            threshold = None

        # Calculate EAR & determine if eye is closed
        eye_closed = 0
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if threshold is not None and ear < threshold:
            eye_closed=1
        print(f"Threshold: {threshold}") # print during testing

        return eye_closed, ear
