from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2

class BlinkProcessor:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def process_ear(self, frame):
        left_eye, right_eye = self.process_face(frame)
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        return avg_ear

    def process_face(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(grey, 0)
        main_face, no_faces = self.extract_main_face(faces)
        if no_faces == 0 or main_face is None:
            return 0, None, None, 0.0
        
        shape = self.predictor(grey, main_face)
        shape = face_utils.shape_to_np(shape)

        left_eye, right_eye = self.extract_eye_regions(shape)
        
        return left_eye, right_eye

    def extract_main_face(self, rects):
        print(f"Number of faces detected: {len(rects)}")
        if not rects:
            return None, None
        return max(rects, key=lambda rect: rect.width() * rect.height()), len(rects)

    def extract_eye_regions(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        return left_eye, right_eye

    @staticmethod
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
