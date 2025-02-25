from scipy.spatial import distance as dist

class BlinkProcessor:
    def __init__(self):
        self.total = 0

    @staticmethod
    def eye_aspect_ratio(eye):
        # A = dist.euclidean(eye[1], eye[5])
        # B = dist.euclidean(eye[2], eye[4])
        # C = dist.euclidean(eye[0], eye[3])
        A = dist.euclidean(eye[0], eye[4])
        B = dist.euclidean(eye[1], eye[3])
        C = dist.euclidean(eye[2], eye[5])
        return (A + B) / (2.0 * C), A, eye[2][0], eye[5][0]

    def process_blink(self, left_eye, right_eye):
        left_ear, A, B, C = self.eye_aspect_ratio(left_eye)
        right_ear, A, B, C = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        return ear, A, B, C
