from scipy.spatial import distance as dist

class BlinkProcessor:
    def __init__(self, eye_ar_thresh=0.24, eye_ar_consec_frames=0):
        self.eye_ar_thresh = eye_ar_thresh
        self.eye_ar_consec_frames = eye_ar_consec_frames
        self.blink_detected = 0
        self.total_blinks = 0

    @staticmethod
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def process_blink(self, left_eye, right_eye):
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        """
        For total blinks, use:
        if ear < self.eye_ar_thresh:
            self.blink_detected += 1
        else:
            if self.blink_detected >= self.eye_ar_consec_frames:
                self.total_blinks += 1
            self.blink_detected = 0
        """

        blink_detected = 1 if ear < self.eye_ar_thresh else 0
        
        return blink_detected, ear
