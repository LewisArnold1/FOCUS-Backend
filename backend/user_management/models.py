import uuid
import os 
from PIL import Image, ImageDraw, ImageFont, ImageOps
import fitz  
from docx import Document
import textwrap


from django.db import models
from django.contrib.auth.models import User
from django.conf import settings


    
def get_unique_file_path(instance, filename):
    ext = filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('documents', unique_filename)

class CalibrationData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    accuracy = models.IntegerField(null=True, blank=True)
    calibration_values = models.JSONField()

    def __str__(self):
        return f"CalibrationData for {self.user} at {self.created_at}, with an accuracy of {self.accuracy}"

class DocumentData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    saved_at = models.DateTimeField(auto_now_add=True)
    file_name = models.CharField(max_length=255)
    file_object = models.FileField(upload_to=get_unique_file_path)
    line_number = models.IntegerField(null=True, blank=True)
    page_number = models.IntegerField(null=True, blank=True)
    favourite = models.BooleanField(default=False)  # Track favorite status

    def generate_preview(self):
        try:
            # Extract filename without extension
            file_base_name = os.path.splitext(os.path.basename(self.file_object.name))[0]
            preview_filename = f"{file_base_name}_preview.jpg"
            
            # Ensure correct path concatenation
            preview_path = os.path.join(settings.MEDIA_ROOT, "documents", preview_filename)

            target_width, target_height = 404, 212  # Landscape: Width fixed at 404
            img_width = None
            img_height = None
            img = None

            if self.file_object.name.endswith(".pdf"):
                with fitz.open(self.file_object.path) as pdf:
                    page = pdf[0]  # Get the first page
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_width, img_height = img.size

            elif self.file_object.name.endswith(".docx"):
                # Extract text from DOCX, keeping paragraphs separate
                document = Document(self.file_object.path)
                paragraphs = [para.text for para in document.paragraphs]  # Include empty paragraphs

                # Create a white background image
                img = Image.new('RGB', (target_width, target_height), color=(255, 255, 255))
                d = ImageDraw.Draw(img)
                img_width, img_height = img.size

                # Load font (ensure arial.ttf is available, or use default)
                try:
                    font = ImageFont.truetype("arial.ttf", 12)  # Adjust size as needed
                except IOError:
                    font = ImageFont.load_default()

                max_chars_per_line = 70  # Adjust for text wrapping
                line_spacing = 5  # Adjust spacing between lines
                x, y = 10, 10  # Start position for text

                for para in paragraphs:
                    if para.strip():  # If paragraph is not empty
                        wrapped_lines = textwrap.wrap(para, width=max_chars_per_line)
                        for wrapped_line in wrapped_lines:
                            # Use textbbox to get the bounding box size
                            bbox = d.textbbox((x, y), wrapped_line, font=font)
                            line_height = bbox[3] - bbox[1]  # Calculate height from bbox
                            if y + line_height > target_height:  # Stop if exceeding image height
                                break

                            # Draw the wrapped text onto the image
                            d.text((x, y), wrapped_line, font=font, fill=(0, 0, 0))
                            y += line_height + line_spacing  # Move cursor down for next line
                    else:
                        # If paragraph is empty, just move down for spacing
                        y += line_spacing  # Keep empty lines with space between paragraphs

            if self.file_object.name.endswith(".txt"):
                # Read the entire file content
                with open(self.file_object.path, 'r') as file:
                    text_preview = file.read()  # Read all text

                # Create a white background image
                img = Image.new('RGB', (target_width, target_height), color=(255, 255, 255))
                d = ImageDraw.Draw(img)
                img_width, img_height = img.size

                # Load font (ensure arial.ttf is available, or use default)
                try:
                    font = ImageFont.truetype("arial.ttf", 12)  # Adjust size as needed
                except IOError:
                    font = ImageFont.load_default()

                max_chars_per_line = 70  # Adjust for text wrapping
                line_spacing = 5  # Adjust spacing between lines
                x, y = 10, 10  # Start position for text

                # Split text by lines and process each line
                lines = text_preview.split("\n")
                for line in lines:
                    # Wrap the non-empty lines that exceed max_chars_per_line
                    if line.strip():  # Only wrap non-empty lines
                        wrapped_lines = textwrap.wrap(line, width=max_chars_per_line)
                        for wrapped_line in wrapped_lines:
                            # Use textbbox to get the bounding box size
                            bbox = d.textbbox((x, y), wrapped_line, font=font)
                            line_height = bbox[3] - bbox[1]  # Calculate height from bbox
                            if y + line_height > target_height:  # Stop if exceeding image height
                                break

                            # Draw the wrapped text onto the image
                            d.text((x, y), wrapped_line, font=font, fill=(0, 0, 0))
                            y += line_height + line_spacing  # Move cursor down for next line
                    else:
                        # Keep empty lines (add an extra line of spacing)
                        y += line_spacing

            if img_width > img_height:
                # Landscape: Resize width to 404 while maintaining aspect ratio
                scale_factor = target_width / img_width
                new_height = int(img_height * scale_factor)
                img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)

                # If height < 212, pad with white
                if new_height < target_height:
                    pad_top = (target_height - new_height) // 2
                    pad_bottom = target_height - new_height - pad_top
                    img = ImageOps.expand(img, (0, pad_top, 0, pad_bottom), fill="white")
                else:
                    # Crop from the top if new height > 212
                    img = img.crop((0, 0, target_width, target_height))
            else:
                # Portrait: Crop top 212 pixels if needed
                # Crop the image along the Y-axis (from top)
                crop_top = 0  # Starting point for Y-axis crop
                crop_bottom = target_height  # Target height to keep

                cropped_img = img.crop((0, crop_top, pix.width, crop_bottom))

                # Resize the cropped image to target dimensions (404x212)
                img = cropped_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            img.save(preview_path)
            return os.path.join("media", "documents", preview_filename).replace("\\", "/")

        except Exception as e:
            print(f"Error generating preview: {e}")
            return None
        
    def __str__(self):
        return f"DocumentData for {self.user} - {self.file_name} at {self.saved_at} which is a favourite: {self.favourite}"
