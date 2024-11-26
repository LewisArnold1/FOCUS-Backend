import json
import base64
from channels.generic.websocket import WebsocketConsumer
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
from eye_processing.blink_detection.count_blinks import process_blink
from datetime import datetime
from channels.exceptions import DenyConnection
import urllib.parse

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
            
            # Generate a new session ID for this video stream
            import random, string
            self.session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            print("Generated session ID:", self.session_id)

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
        text_data_json = json.loads(text_data)
        frame_data = text_data_json.get('frame', None)
        timestamp = text_data_json.get('timestamp', None)  # Extract timestamp

        if frame_data:
            # Process the frame and get the blink count
            self.process_frame(frame_data, timestamp)

    def process_frame(self, frame_data, timestamp):
        try:
            from eye_processing.models import SimpleEyeMetrics
            # Decode the base64-encoded image
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Call the blink detection function with the frame
            total_blinks, ear = process_blink(frame)

            # Print or send the results (e.g., to the frontend or console)
            # Convert the timestamp from milliseconds to a datetime object
            timestamp_s = timestamp / 1000
            timestamp_dt = datetime.fromtimestamp(timestamp_s)

             # Save the metrics for this frame in the database with the user
            eye_metrics = SimpleEyeMetrics(
                user=self.user,  # Associate the logged-in user
                session_id=self.session_id, # Associate current sessionID
                timestamp=timestamp_dt,
                blink_count=total_blinks,
                eye_aspect_ratio=ear,   
            )
            print(self.session_id)
            eye_metrics.save()

            print(f"User: {self.user.username}, Timestamp: {timestamp_dt}, Total Blinks: {total_blinks}, EAR: {ear}")
        except (base64.binascii.Error, UnidentifiedImageError) as e:
            print("Error decoding image:", e)