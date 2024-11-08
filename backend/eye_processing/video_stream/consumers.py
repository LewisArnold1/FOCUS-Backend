import json
import base64
import os
from channels.generic.websocket import WebsocketConsumer
from io import BytesIO
from PIL import Image

class VideoFrameConsumer(WebsocketConsumer):
    counter = 0  # Initialise the counter - used to name image file for each frame

    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        self.counter = 0 # Reset counter on disconnection

    def receive(self, text_data):
        # Parse the received JSON message
        text_data_json = json.loads(text_data)
        frame_data = text_data_json.get('frame', None)

        if frame_data:
            self.save_frame(frame_data)

        print("Received frame")

    def save_frame(self, frame_data):
        # Decode the base64-encoded image
        image_data = base64.b64decode(frame_data.split(',')[1])  # Split to remove 'data:image/jpeg;base64,'

        # Create a PIL Image from the decoded bytes
        image = Image.open(BytesIO(image_data))

        # Define the directory where images will be saved
        save_dir = 'received_frames'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate a unique filename using the counter
        filename = os.path.join(save_dir, f'frame_{self.counter}.jpg')
        
        # Save the image
        image.save(filename)

        print(f"Saved frame to {filename}")

        # Increment the counter after saving
        self.counter += 1
