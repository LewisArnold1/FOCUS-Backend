from scipy.spatial import distance as dist

class BlinkProcessor:
    def __init__(self, eye_ar_thresh=0.25, eye_ar_consec_frames=2):
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

    def process_blink(self, left_eye, right_eye):
        eye_closed = 0
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < self.eye_ar_thresh:
            eyes_closed=1

        return eye_closed, ear
