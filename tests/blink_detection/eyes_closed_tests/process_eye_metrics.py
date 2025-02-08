import os

from .face import FaceProcessor
from .blinks import BlinkProcessor
from .pupil import PupilProcessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(CURRENT_DIR, 'shape_predictor_68_face_landmarks.dat')
face_processor = FaceProcessor(PREDICTOR_PATH)
blink_processor = BlinkProcessor()
pupil_processor = PupilProcessor()


'''
Method needs two arguments while testing auto blink threshold
'''
def process_eye(frame):               # for Most
# def process_eye(frame, ear_list):       # For auto only

    
    # Extract left and right eye landmarks
    left_eye, right_eye = face_processor.process_face(frame)
    if left_eye is None or right_eye is None:
        print("No eye")
        return 0, None, None
    
    # Process pupil coordinates
    # pupil = pupil_processor.process_pupil(frame, left_eye)
    pupil = None
    
    '''
    Uncomment blink test methods as required.
    Also change what is returned as required.
    '''
    
    '''
    Manual
    '''
    closed, ear = blink_processor.manual_threshold(left_eye, right_eye)

    '''
    Auto - for now requires ear_list as argument. Unnecessary in website
    '''
    # closed, ear = blink_processor.auto_threshold(left_eye, right_eye, ear_list)


    '''
    CNN
    '''
    # closed = blink_processor.CNN(left_eye, right_eye)

    return closed, ear, pupil
