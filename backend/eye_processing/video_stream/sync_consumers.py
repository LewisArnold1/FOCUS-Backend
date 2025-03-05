# Synchronous (single-process) implementation of the WebSocket consumer for video streaming
#
# This uses `daphne` to run the ASGI application
#
# daphne backend.asgi:application

import base64
import json
import urllib.parse
from datetime import datetime, timedelta
from io import BytesIO
import os
import django
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import base64

from channels.generic.websocket import WebsocketConsumer
from django.db.models import Max

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()  # Ensure Django is initialised before importing Django modules

from eye_processing.models import SimpleEyeMetrics, UserSession
from eye_processing.eye_metrics.process_eye_metrics import process_eye
from eye_processing.eye_metrics.process_blinks import process_ears

from asgiref.sync import sync_to_async
from rest_framework_simplejwt.authentication import JWTAuthentication


def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def decode_frame(encoded_frame):
    buffer = np.frombuffer(base64.b64decode(encoded_frame), dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

TIME_WINDOW = 0.5

class VideoFrameConsumer(WebsocketConsumer):

    ## Performance testing
    total_frames = 0
    # frames = []  # Store frames for blink detection
    # latencies = []  # Store latencies for each frame

    ear_list = [] # List to store EAR values for adaptive thresholding

    def connect(self):
        query_string = self.scope['query_string'].decode('utf-8')
        print("Query string received:", query_string)

        try:
            encoded_token_data = query_string.split('=')[1]
            decoded_token_data = urllib.parse.unquote(encoded_token_data)
            token_data = json.loads(decoded_token_data)

            self.token = token_data.get("access", None)
            if not self.token:
                raise ValueError("Access token not found in query string")

            print("Extracted token:", self.token)

            # Run authentication in a synchronous thread
            validated_token = JWTAuthentication().get_validated_token(self.token)
            self.user = JWTAuthentication().get_user(validated_token)

            # Fetch max video_id in an async-safe way
            max_session_id = UserSession.objects.filter(user=self.user).aggregate(Max('session_id'))
            max_video_id = SimpleEyeMetrics.objects.filter(user=self.user, session_id=max_session_id['session_id__max']).aggregate(Max('video_id'))

            self.video_id = (max_video_id['video_id__max'] or 0) + 1
            self.session_id = max_session_id['session_id__max']  

            self.accept()
        except IndexError:
            print("Invalid query string format:", query_string)
            self.close()
        except Exception as e:
            print("Authentication failed:", e)
            self.close()
            
    def disconnect(self, close_code):
        ## Performance testing
        # print(self.frames)
        # print(self.latencies)
        # self.total_frames = 0
        # self.frames.clear()
        # self.latencies.clear()
        self.close()

    def receive(self, text_data):
        try:
            # Parse the received JSON message
            data_json = json.loads(text_data)
            frame_data = data_json.get('frame', None)
            timestamp = data_json.get('timestamp', None)  # Extract timestamp
            mode = data_json.get('mode', 'reading')  # Default to 'reading' if not provided
            reading_mode = data_json.get('reading_mode', 3)
            wpm = data_json.get('wpm', 0)

            ## Performance testing
            self.total_frames = self.total_frames + 1
            if(self.total_frames % 30 == 0):
                print("Total Frames: ", self.total_frames)
                latency = datetime.now() - datetime.fromtimestamp(timestamp/1000)
                print("Latency: ", latency)
            #     self.frames.append(self.total_frames)
            #     self.latencies.append(str(latency))

            if mode == "reading":
                x_coordinate_px = data_json.get('xCoordinatePx', None)
                y_coordinate_px = data_json.get('yCoordinatePx', None)

                if frame_data:
                    self.process_reading_frame(frame_data, timestamp, x_coordinate_px, y_coordinate_px, reading_mode, wpm)

            elif mode == "diagnostic":
                draw_mesh = data_json.get('draw_mesh', False)
                draw_contours = data_json.get('draw_contours', False)
                show_axis = data_json.get('show_axis', False) 
                draw_eye = data_json.get('draw_eye', False)

                if frame_data:
                    self.process_diagnostic_frame(frame_data, timestamp, draw_mesh, draw_contours, show_axis, draw_eye)
        except Exception as e:
            print("Error processing frame:", e)
            self.disconnect(1000)

    def process_reading_frame(self, frame_data, timestamp, x_coordinate_px, y_coordinate_px, reading_mode, wpm):
        try:
            # print("PROCESSING frame no. ", self.total_frames)

            # Decode the incoming frame
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert timestamp
            timestamp_s = timestamp / 1000
            timestamp_dt = datetime.fromtimestamp(timestamp_s)

            avg_ear = process_ears(frame) # Process EAR value for the current frame

            blink_detected=False

            ############################################################################# Adaptive thresholding for blink detection
            threshold = 0.0

            if(avg_ear != None):
                if len(self.ear_list) < 30:
                    self.ear_list.append(avg_ear)
                else:
                    # calculate the maximum EAR value in the list
                    max_ear = max(self.ear_list)
                    threshold = max_ear * 0.7
                            
                # Adaptive thresholding for blink detection
                if avg_ear < threshold and threshold != 0.0: # Blinks are only detected after the first 30 frames, in order to accurately calculate the correct threshold
                    blink_detected = True
                    # print("Blink detected at frame: ", self.total_frames)
                else:
                    blink_detected = False
                    # print("Blink not detected"

            #############################################################################

            # Process the 
            face_detected, normalised_eye_speed, yaw, pitch, roll, left_centre, right_centre, focus, left_iris_velocity, right_iris_velocity, movement_type, _ = process_eye(frame, timestamp_dt, blink_detected)

            # Extract eye metrics
            max_session_id = UserSession.objects.filter(user=self.user).aggregate(Max('session_id'))
            session_id = max_session_id['session_id__max']

             # Save the metrics for this frame in the database with the user
            eye_metrics = SimpleEyeMetrics(
                user=self.user,  # Associate the logged-in user
                session_id=session_id,
                video_id=self.video_id, # Associate current videoID
                timestamp=timestamp_dt,
                gaze_x=x_coordinate_px,
                gaze_y=y_coordinate_px,
                face_detected=face_detected,
                normalised_eye_speed=normalised_eye_speed,
                face_yaw=yaw,
                face_roll=roll,
                face_pitch=pitch,
                eye_aspect_ratio=avg_ear,
                blink_detected=blink_detected,
                left_centre=left_centre, 
                right_centre=right_centre,
                focus=focus,
                left_iris_velocity=left_iris_velocity,
                right_iris_velocity=right_iris_velocity, 
                movement_type=movement_type,
            )
            eye_metrics.save()

            # print("FINISHED frame no. ", self.total_frames)

            ############################################################################# SVM classification for blink detection - No longer working under new FPS constraints (requires 30 FPS)
            # Get past frames within time_window * 2
            # start_time = timestamp_dt - timedelta(seconds=TIME_WINDOW * 2)

            # # Check if at least one frame exists before start_time
            # frame_before_window = await sync_to_async(SimpleEyeMetrics.objects.filter(
            #     user=self.user, session_id=session_id,
            #     video_id=self.video_id,
            #     timestamp__lt=start_time
            # ).exists)()

            # middle_frame = None

            # if frame_before_window:

            #     # Retrieve frames in the time window
            #     past_frames = await sync_to_async(list)(SimpleEyeMetrics.objects.filter(
            #         user=self.user, session_id=session_id,
            #         video_id=self.video_id,
            #         timestamp__gte=start_time,
            #         timestamp__lte=timestamp_dt
            #     ).order_by("timestamp"))

            #     # Extract middle frame 
            #     middle_index = len(past_frames) // 2
            #     middle_frame_entry = past_frames[middle_index] if past_frames else None

            #     if middle_frame_entry and middle_frame_entry.frame:
            #         # Decode middle frame
            #         middle_frame = decode_frame(middle_frame_entry.frame)

            #         # Get EAR values for the full window
            #         ear_values = [entry.eye_aspect_ratio for entry in past_frames]
            #         timestamps = [entry.timestamp for entry in past_frames]
            #         middle_frame_timestamp = middle_frame_entry.timestamp
                    
            #         # Add blink detection processing to the async task list
            #         if ear_values and timestamps:
            #             task = asyncio.create_task(asyncio.to_thread(process_blinks, ear_values, timestamps, middle_frame_timestamp))
            #             self.tasks.append(task)

            #             # Await the blink detection result
            #             blink_detected = await task
            #         else:
            #             blink_detected = False

            #         if middle_frame is not None:
            #             face_detected, normalised_eye_speed, yaw, pitch, roll, left_centre, right_centre, focus, left_iris_velocity, right_iris_velocity, movement_type, _ = process_eye(middle_frame, middle_frame_entry.timestamp, blink_detected)

            #             # Update database for the middle frame
            #             await sync_to_async(SimpleEyeMetrics.objects.filter(
            #                 user=self.user, session_id=session_id,
            #                 video_id=self.video_id,
            #                 timestamp=middle_frame_entry.timestamp
            #             ).update)(
            #                 face_detected=face_detected,
            #                 normalised_eye_speed=normalised_eye_speed,
            #                 face_yaw=yaw,
            #                 face_roll=roll,
            #                 face_pitch=pitch,
            #                 left_centre=left_centre,
            #                 right_centre=right_centre,
            #                 focus=focus,
            #                 left_iris_velocity=left_iris_velocity,
            #                 right_iris_velocity=right_iris_velocity,
            #                 movement_type=movement_type,
            #                 blink_detected=blink_detected
            #             )

            #             # Cleanup: Delete old frames outside of time_window * 2
            #             await sync_to_async(lambda: SimpleEyeMetrics.objects.filter(
            #                 user=self.user, 
            #                 session_id=session_id,
            #                 video_id=self.video_id,
            #                 timestamp__lt=start_time
            #             ).update(frame=None))()

            #############################################################################

        except (base64.binascii.Error, UnidentifiedImageError) as e:
            print("Error decoding image:", e)

    def process_diagnostic_frame(self, frame_data, timestamp, draw_mesh, draw_contours, show_axis, draw_eye):
        try:
            # Decode the base64-encoded image
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert the timestamp from milliseconds to a datetime object
            timestamp_s = timestamp / 1000
            timestamp_dt = datetime.fromtimestamp(timestamp_s)

            # Call `process_eye` with visualisation options
            face_detected, normalised_eye_speed, yaw, pitch, roll, left_centre, right_centre, focus, left_iris_velocity, right_iris_velocity, movement_type, diagnostic_frame = process_eye(frame, timestamp_dt, blink_detected=False, draw_mesh=draw_mesh, draw_contours=draw_contours, show_axis=show_axis, draw_eye=draw_eye)

            # Encode the processed frame back to base64
            _, buffer = cv2.imencode('.jpg', diagnostic_frame)
            processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Send processed image back via WebSocket
            self.send(text_data=json.dumps({
            "mode": "diagnostic",
            "frame": f"data:image/jpeg;base64,{processed_frame_base64}",
            "face_detected": face_detected,
            "yaw": float(yaw) if yaw != None else None,
            "pitch": float(pitch) if pitch != None else None,
            "roll": float(roll) if roll != None else None,
            "eye_speed": float(normalised_eye_speed) if normalised_eye_speed != None else None,
            }))

        except (base64.binascii.Error, UnidentifiedImageError) as e:
            print("Error decoding image:", e)

