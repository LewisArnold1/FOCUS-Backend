from django.db import models
from django.contrib.auth.models import User

class SimpleEyeMetrics(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)  # Track the logged-in user
    timestamp = models.DateTimeField()  # Store the timestamp for each frame
    blink_count = models.IntegerField()  # Store the blink count
    eye_aspect_ratio = models.FloatField(null=True, blank=True)  # Store the eye aspect ratio
    x_coordinate_px = models.FloatField(null=True, blank=True)
    y_coordinate_px = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"User: {self.user.username} - Timestamp: {self.timestamp} - Blinks: {self.blink_count} - EAR: {self.eye_aspect_ratio}"