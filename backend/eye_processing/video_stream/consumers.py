import json
import base64
from channels.generic.websocket import WebsocketConsumer
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from eye_processing.blink_detection.count_blinks import process_blink
from datetime import datetime

class VideoFrameConsumer(WebsocketConsumer):

    def connect(self):
        self.accept()

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
            print(f"Timestamp: {timestamp_dt}, Total Blinks: {total_blinks}, EAR: {ear}")

            # Save the metrics for this frame in the database
            eye_metrics = SimpleEyeMetrics(timestamp=timestamp_dt, blink_count=total_blinks, eye_aspect_ratio=ear,)
            eye_metrics.save()
        except (base64.binascii.Error, PIL.UnidentifiedImageError) as e:
            print("Error decoding image:", e)