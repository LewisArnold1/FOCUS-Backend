import uuid
import os 
from PIL import Image, ImageDraw, ImageFont
import fitz  
from docx import Document


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

            if self.file_object.name.endswith(".pdf"):
                # PDF preview generation with cropping along Y-axis and resizing (no borders)
                with fitz.open(self.file_object.path) as pdf:
                    page = pdf[0]  # Get the first page
                    
                    # Generate the pixmap for the PDF page (no zooming)
                    pix = page.get_pixmap()

                    # Convert the pixmap to a PIL image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Define the target dimensions (404 x 212)
                    target_width, target_height = 404, 212

                    # Crop the image along the Y-axis (from top)
                    crop_top = 0  # Starting point for Y-axis crop
                    crop_bottom = target_height  # Target height to keep

                    cropped_img = img.crop((0, crop_top, pix.width, crop_bottom))

                    # Resize the cropped image to target dimensions (404x212)
                    resized_img = cropped_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                    # Save the resized image directly
                    resized_img.save(preview_path)
                    return os.path.join("media", "documents", preview_filename).replace("\\", "/")  # Ensures correct format

            elif self.file_object.name.endswith(".docx"):
                # DOCX preview generation with cropping along Y-axis and resizing (no borders)
                document = Document(self.file_object.path)
                text_preview = "\n".join([para.text for para in document.paragraphs])  # All paragraphs combined

                # Create an image with Pillow for preview
                img = Image.new('RGB', (612, 792), color=(255, 255, 255))  # White background
                d = ImageDraw.Draw(img)
                
                # Set font (Optional, you can choose another font path or size)
                try:
                    font = ImageFont.truetype("arial.ttf", 36)  # Make sure you have arial.ttf installed
                except IOError:
                    font = ImageFont.load_default()  # Default font if arial is not found

                # Draw the text onto the image
                d.text((10, 10), text_preview, font=font, fill=(0, 0, 0))  # Black text

                # Now scale the image while maintaining the aspect ratio
                target_width, target_height = 404, 212
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height

                # Calculate new dimensions
                if img_width > img_height:
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)

                # Resize the image while maintaining the aspect ratio
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Crop the image along the Y-axis (from top) for DOCX
                crop_top = 0  # Starting point for Y-axis crop
                crop_bottom = target_height  # Target height to keep

                cropped_img = img.crop((0, crop_top, new_width, crop_bottom))

                # Resize the cropped image to target dimensions (404x212)
                final_img = cropped_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                # Save the final image directly (without borders)
                final_img.save(preview_path)
                return os.path.join("media", "documents", preview_filename).replace("\\", "/")  # Ensures correct format

            elif self.file_object.name.endswith(".txt"):
                # TXT preview generation with cropping along Y-axis and resizing (no borders)
                with open(self.file_object.path, 'r') as file:
                    text_preview = file.read()  # Read the entire file content

                # Create an image with Pillow for preview
                img = Image.new('RGB', (612, 792), color=(255, 255, 255))  # White background
                d = ImageDraw.Draw(img)

                # Set font (Optional, you can choose another font path or size)
                try:
                    font = ImageFont.truetype("arial.ttf", 36)  # Make sure you have arial.ttf installed
                except IOError:
                    font = ImageFont.load_default()  # Default font if arial is not found

                # Draw the text onto the image
                d.text((10, 10), text_preview, font=font, fill=(0, 0, 0))  # Black text

                # Now scale the image while maintaining the aspect ratio
                target_width, target_height = 404, 212
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height

                # Calculate new dimensions
                if img_width > img_height:
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)

                # Resize the image while maintaining the aspect ratio
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Crop the image along the Y-axis (from top) for TXT
                crop_top = 0  # Starting point for Y-axis crop
                crop_bottom = target_height  # Target height to keep

                cropped_img = img.crop((0, crop_top, new_width, crop_bottom))

                # Resize the cropped image to target dimensions (404x212)
                final_img = cropped_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                # Save the final image directly (without borders)
                final_img.save(preview_path)
                return os.path.join("media", "documents", preview_filename).replace("\\", "/")  # Ensures correct format

            return None

        except Exception as e:
            print(f"Error generating preview: {e}")
            return None
        
    def __str__(self):
        return f"DocumentData for {self.user} - {self.file_name} at {self.saved_at} which is a favourite: {self.favourite}"
