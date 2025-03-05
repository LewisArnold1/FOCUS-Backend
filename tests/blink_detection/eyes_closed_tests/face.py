import cv2
import dlib
from imutils import face_utils

class FaceProcessor:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.prev_center = None  
        self.prev_time = None  

    def extract_main_face(self, rects):
        # print(f"Number of faces detected: {len(rects)}")
        if not rects:
            return None, None
        return max(rects, key=lambda rect: rect.width() * rect.height()), len(rects)

    def extract_eye_regions(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        return left_eye, right_eye
    
    def process_face(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(grey, 0)
        main_face, no_faces = self.extract_main_face(faces)
        if no_faces == 0 or main_face is None:
            return 0, None, None, 0.0
        
        shape = self.predictor(grey, main_face)
        shape = face_utils.shape_to_np(shape)

        left_eye, right_eye = self.extract_eye_regions(shape)
        
        return no_faces, left_eye, right_eye
