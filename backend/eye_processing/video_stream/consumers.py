import base64
import json
import urllib.parse
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from channels.generic.websocket import WebsocketConsumer

from django.db.models import Max

from eye_processing.eye_metrics import process_eye

class VideoFrameConsumer(WebsocketConsumer):

    def connect(self):
        query_string = self.scope['query_string'].decode('utf-8')
        print("Query string received:", query_string)  # Debugging log

        try:
            # Split by '=' to extract the token content
            encoded_token_data = query_string.split('=')[1]
            
            # Decode URL encoding (e.g., %22 -> ")
            decoded_token_data = urllib.parse.unquote(encoded_token_data)
            print("Decoded token data:", decoded_token_data)

            # Parse JSON content
            token_data = json.loads(decoded_token_data)
            print("Parsed token data:", token_data)

            # Extract the "access" token
            self.token = token_data.get("access", None)
            if not self.token:
                raise ValueError("Access token not found in query string")

            print("Extracted token:", self.token)
            from rest_framework_simplejwt.authentication import JWTAuthentication
            validated_token = JWTAuthentication().get_validated_token(self.token)
            self.user = JWTAuthentication().get_user(validated_token)
            
            # increment max video id for this user
            from eye_processing.models import SimpleEyeMetrics
            from eye_processing.models import UserSession
            # filter by user & session
            max_video_id = SimpleEyeMetrics.objects.filter(user=self.user,session_id=UserSession.objects.filter(user=self.user).aggregate(Max('session_id'))['session_id__max']).aggregate(Max('video_id'))['video_id__max'] or 0
            self.video_id = max_video_id + 1

            self.accept()
        except IndexError:
            print("Invalid query string format:", query_string)
            self.close()
        except Exception as e:
            print("Authentication failed:", e)
            self.close()
            
    def disconnect(self, close_code):
        pass 

    def receive(self, text_data):
        # Parse the received JSON message
        data_json = json.loads(text_data)
        frame_data = data_json.get('frame', None)
        timestamp = data_json.get('timestamp', None)  # Extract timestamp
        x_coordinate_px = data_json.get('xCoordinatePx', None)
        y_coordinate_px = data_json.get('yCoordinatePx', None)

        if frame_data:
            # Process the frame and get the blink count
            self.process_frame(frame_data, timestamp, x_coordinate_px, y_coordinate_px)

    def process_frame(self, frame_data, timestamp, x_coordinate_px, y_coordinate_px):
        try:
            from eye_processing.models import SimpleEyeMetrics
            # Decode the base64-encoded image
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Extract eye metrics
            no_faces, normalised_face_speed, ear, blink_detected, left_centre, right_centre = process_eye(frame)

            # Convert the timestamp from milliseconds to a datetime object
            timestamp_s = timestamp / 1000
            timestamp_dt = datetime.fromtimestamp(timestamp_s)

             # Save the metrics for this frame in the database with the user
            from eye_processing.models import UserSession
            eye_metrics = SimpleEyeMetrics(
                user=self.user,  # Associate the logged-in user
                session_id=UserSession.objects.filter(user=self.user).aggregate(Max('session_id'))['session_id__max'],
                video_id=self.video_id, # Associate current videoID
                timestamp=timestamp_dt,
                x_coordinate_px = x_coordinate_px,
                y_coordinate_px = y_coordinate_px,
                no_faces=no_faces,
                normalised_face_speed=normalised_face_speed,
                eye_aspect_ratio=ear,
                blink_count=blink_detected,
                left_centre=left_centre, 
                right_centre=right_centre
            )
            eye_metrics.save()

            print(f"User: {self.user.username}, Timestamp: {timestamp_dt}, Total Blinks: {blink_detected}, EAR: {ear}, x-coordinate: {x_coordinate_px}, y-coordinate: {y_coordinate_px}, Session ID: {eye_metrics.session_id}, Video ID: {eye_metrics.video_id}")
        except (base64.binascii.Error, UnidentifiedImageError) as e:
            print("Error decoding image:", e)