import cv2
import numpy as np

def crop_eyes_with_space(image_path):
    image = cv2.imread(image_path)# Load the color image
    if image is None:
        print("Error loading image.") #return errror 
        return None
    
    # Initialize face and eye detectors
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    #grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print("No faces detected.")
        return None

    for (x, y, w, h) in faces:
        
        face_region_gray = gray[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_region_gray)
        
        # Sort eyes by x-coordinate to find left and right eye
        eyes = sorted(eyes, key=lambda e: e[0])
        
        if len(eyes) >= 2:
            (ex1, ey1, ew1, eh1) = eyes[0]  # Left eye
            (ex2, ey2, ew2, eh2) = eyes[1]  # Right eye

            # Define cropping boundaries
            top = y + min(ey1, ey2)
            bottom = y + max(ey1 + eh1, ey2 + eh2)
            left = x + ex1
            right = x + ex2 + ew2

        elif len(eyes) == 1:
            # If only one eye is detected, crop around it with buffer space
            (ex, ey, ew, eh) = eyes[0]
            buffer = 20  # Optional buffer space around the single eye
            top = max(y + ey - buffer, 0)
            bottom = min(y + ey + eh + buffer, image.shape[0])
            left = max(x + ex - buffer, 0)
            right = min(x + ex + ew + buffer, image.shape[1])

        else:
            # No eyes detected
            print("Eyes were not detected.")
            return None

        # Crop resize the eye space region
        eye_space_region = image[top:bottom, left:right]
        
        # Resize
        target_height = 50 #change accordingly
        aspect_ratio = (right - left) / (bottom - top)
        target_width = int(target_height * aspect_ratio)
        eye_space_resized = cv2.resize(eye_space_region, (target_width, target_height))
        
        # Save and display 
        cv2.imwrite("cropped_eyes_with_space.jpg", eye_space_resized)
        print("Cropped eyes with space saved as 'cropped_eyes_with_space.jpg'")
        
        cv2.imshow("Cropped Eyes with Space", eye_space_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return eye_space_resized

    print("Eyes were not detected.")
    return None
