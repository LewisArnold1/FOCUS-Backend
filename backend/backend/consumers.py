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

        print("Received frame")

        if frame_data:
            self.save_frame(frame_data)

    def save_frame(self, frame_data):
        # Decode the base64-encoded image
        image_data = base64.b64decode(frame_data.split(',')[1])  # Split to remove 'data:image/jpeg;base64,'

        # Create a PIL Image from the decoded bytes
        image = Image.open(BytesIO(image_data))

        # Define the directory where images will be saved
        save_dir = 'received_frames'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate a unique filename (you can use a timestamp or counter for uniqueness)
        filename = os.path.join(save_dir, 'frame_{}.jpg'.format(self.get_next_filename(save_dir)))
        
        # Save the image
        image.save(filename)
        print(f"Saved frame to {filename}")

    def get_next_filename(self, save_dir):
        # Count the number of images in the directory to get the next number
        existing_files = os.listdir(save_dir)
        image_files = [f for f in existing_files if f.endswith('.jpg')]
        return len(image_files)
