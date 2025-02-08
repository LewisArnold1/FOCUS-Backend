import uuid, os 
from django.db import models
from django.contrib.auth.models import User

    
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
    line_number = models.IntegerField()
    page_number = models.IntegerField()
    timestamp = models.DateTimeField()

    def __str__(self):
        return f"DocumentData for {self.user} - {self.file_name} at {self.saved_at}"
