import json
import base64
from channels.generic.websocket import WebsocketConsumer
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from eye_processing.blink_detection.count_blinks import process_blink

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
        # Decode the base64-encoded image
        image_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Call the blink detection function with the frame
        total_blinks, ear = process_blink(frame)

        # Print or send the results (e.g., to the frontend or console)
        print(f"Timestamp: {timestamp}, Total Blinks: {total_blinks}, EAR: {ear}")

        # If you want to send results back to the frontend
        '''
        self.send(text_data=json.dumps({
            'timestamp': timestamp,
            'total_blinks': total_blinks,
            'ear': ear
        
        }))
        '''