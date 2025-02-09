import uuid
import os 
from PIL import Image
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
        """
        Generate a preview image for the file (PDF, DOC/DOCX).
        """
        try:
            # Define the path where the preview will be saved
            preview_filename = f"{os.path.splitext(self.file_object.name)[0]}_preview.jpg"
            preview_path = os.path.join(settings.MEDIA_ROOT, 'documents', preview_filename)

            # Handle PDF files
            if self.file_object.name.endswith('.pdf'):
                with fitz.open(self.file_object.path) as pdf:
                    # Extract the first page
                    page = pdf[0]
                    pix = page.get_pixmap()
                    pix.save(preview_path)
                    return f"/media/documents/{preview_filename}"  # Relative path for serving

            # Handle DOC/DOCX files
            elif self.file_object.name.endswith(('.doc', '.docx')):
                document = Document(self.file_object.path)
                text_preview = "\n".join([paragraph.text for paragraph in document.paragraphs[:3]])
                text_preview_path = os.path.join(settings.MEDIA_ROOT, 'documents', f"{os.path.splitext(self.file_object.name)[0]}_preview.txt")
                with open(text_preview_path, "w") as file:
                    file.write(text_preview)
                return f"/media/documents/{os.path.basename(text_preview_path)}"  # Relative path for serving

            # No preview for unsupported types
            return None

        except Exception as e:
            print(f"Error generating preview: {e}")
            return None

    def __str__(self):
        return f"DocumentData for {self.user} - {self.file_name} at {self.saved_at}"
