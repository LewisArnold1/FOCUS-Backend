import json
import base64
import os
from channels.generic.websocket import WebsocketConsumer
from io import BytesIO
from PIL import Image

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
            self.save_frame(frame_data, timestamp)

        print(f"Received frame with timestamp: {timestamp}")

    def save_frame(self, frame_data, timestamp):
        # Decode the base64-encoded image
        image_data = base64.b64decode(frame_data.split(',')[1])  # Split to remove 'data:image/jpeg;base64,'

        # Create a PIL Image from the decoded bytes
        image = Image.open(BytesIO(image_data))

        # Define the directory where images will be saved
        save_dir = 'received_frames'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate a unique filename using the timestamp
        filename = os.path.join(save_dir, f'frame_{timestamp}.jpg')
        
        # Save the image
        image.save(filename)

        print(f"Saved frame to {filename} with timestamp {timestamp}")
