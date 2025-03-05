from scipy.spatial import distance as dist

class BlinkProcessor:
    def __init__(self):
        pass

    def eye_aspect_ratio(self, left_eye, right_eye):
        # Left Eye
        A = dist.euclidean(left_eye[1], left_eye[5])
        B = dist.euclidean(left_eye[2], left_eye[4])
        C = dist.euclidean(left_eye[0], left_eye[3])
        left_EAR = (A + B) / (2.0 * C)
        
        # Right Eye
        A = dist.euclidean(right_eye[1], right_eye[5])
        B = dist.euclidean(right_eye[2], right_eye[4])
        C = dist.euclidean(right_eye[0], right_eye[3])
        right_EAR = (A + B) / (2.0 * C)

        avg_EAR = (left_EAR + right_EAR) / 2.0

        return avg_EAR
    

    