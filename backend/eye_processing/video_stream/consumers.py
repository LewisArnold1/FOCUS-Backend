import base64
import json
import urllib.parse
from datetime import datetime
from io import BytesIO
import os
import django

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from channels.generic.websocket import AsyncWebsocketConsumer

from django.db.models import Max

from eye_processing.eye_metrics.process_eye_metrics import process_eye

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()  # Ensure Django is initialized before importing Django modules

from asgiref.sync import sync_to_async
from rest_framework_simplejwt.authentication import JWTAuthentication

class VideoFrameConsumer(AsyncWebsocketConsumer):

    async def connect(self):
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
            validated_token = await sync_to_async(JWTAuthentication().get_validated_token)(self.token)
            self.user = await sync_to_async(JWTAuthentication().get_user)(validated_token)

            # Fetch max video_id in an async-safe way
            from eye_processing.models import SimpleEyeMetrics, UserSession

            max_session_id = await sync_to_async(UserSession.objects.filter(user=self.user).aggregate)(Max('session_id'))
            max_video_id = await sync_to_async(SimpleEyeMetrics.objects.filter(user=self.user, session_id=max_session_id['session_id__max']).aggregate)(Max('video_id'))

            self.video_id = (max_video_id['video_id__max'] or 0) + 1

            await self.accept()
        except IndexError:
            print("Invalid query string format:", query_string)
            await self.close()
        except Exception as e:
            print("Authentication failed:", e)
            await self.close()
            
    async def disconnect(self, close_code):
        pass 

    async def receive(self, text_data):
        # Parse the received JSON message
        data_json = json.loads(text_data)
        frame_data = data_json.get('frame', None)
        timestamp = data_json.get('timestamp', None)  # Extract timestamp
        x_coordinate_px = data_json.get('xCoordinatePx', None)
        y_coordinate_px = data_json.get('yCoordinatePx', None)

        if frame_data:
            # Process the frame and get the blink count                
            await self.process_frame(frame_data, timestamp, x_coordinate_px, y_coordinate_px)

    async def process_frame(self, frame_data, timestamp, x_coordinate_px, y_coordinate_px):
        try:
            from eye_processing.models import SimpleEyeMetrics, UserSession
            # Decode the base64-encoded image
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Extract eye metrics
            face_detected, normalised_eye_speed, yaw, pitch, roll, avg_ear, blink_detected, left_centre, right_centre, focus = process_eye(frame)

            # Convert the timestamp from milliseconds to a datetime object
            timestamp_s = timestamp / 1000
            timestamp_dt = datetime.fromtimestamp(timestamp_s)

            max_session_id = await sync_to_async(UserSession.objects.filter(user=self.user).aggregate)(Max('session_id'))
            session_id = max_session_id['session_id__max']

             # Save the metrics for this frame in the database with the user
            eye_metrics = SimpleEyeMetrics(
                user=self.user,  # Associate the logged-in user
                session_id=session_id,
                video_id=self.video_id, # Associate current videoID
                timestamp=timestamp_dt,
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
            )
            await sync_to_async(eye_metrics.save)()

            print(f"User: {self.user.username}, Timestamp: {timestamp_dt}, Total Blinks: {blink_detected}, EAR: {ear}, x-coordinate: {x_coordinate_px}, y-coordinate: {y_coordinate_px}, Session ID: {eye_metrics.session_id}, Video ID: {eye_metrics.video_id}")
        except (base64.binascii.Error, UnidentifiedImageError) as e:
            print("Error decoding image:", e)