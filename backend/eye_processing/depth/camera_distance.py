import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

# cap = cv2.VideoCapture(0)
# detector = FaceMeshDetector(maxFaces=1)
# # make first argument a frame , cap and detector inside func

# def calculate_depth(frame):
#     # return depth_cm
#     pass

# # process one frame at a time 
# # 

# while True:
#     success, img = cap.read()
#     img, faces = detector.findFaceMesh(img, draw=False)

#     if faces:
#         face = faces[0]
#         pointLeft = face[145]
#         pointRight = face[374]
#         # Drawing
#         # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
#         # cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
#         # cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
#         w, _ = detector.findDistance(pointLeft, pointRight)
#         W = 6.3

#         # # Finding the Focal Length
#         # d = 50
#         # f = (w*d)/W
#         # print(f)

#         # Finding distance
#         f = 840
#         d = (W * f) / w
#         print(d)

#         cvzone.putTextRect(img, f'Depth: {int(d)}cm',
#                            (face[10][0] - 100, face[10][1] - 50),
#                            scale=2)

   
#     cv2.imshow("Image", img)
#     # Check for key press
#     key = cv2.waitKey(1)
#     if key == ord('q'):  # Press 'q' to quit
#         break

# # Release the camera and close windows
# cap.release()
# cv2.destroyAllWindows()


# Initialize the FaceMeshDetector
detector = FaceMeshDetector(maxFaces=1)

def calculate_depth(frame):
    cap = cv2.VideoCapture(0)
    # Process one frame at a time
    success, img = cap.read()
    img, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        
        # Find the distance between the two points
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3  # Real-world width of the face
        
        # Focal length (can be calibrated based on real-world testing)
        f = 840
        
        # Calculate the distance from the camera
        d = (W * f) / w

        # Return the depth value
        return d
    else:
        # Return None if no face is detected
        return None