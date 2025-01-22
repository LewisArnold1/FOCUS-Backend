from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField

class UserSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)  # Track the logged-in user
    session_id = models.IntegerField(default=0)  # Track login session

    def __str__(self):
        return f"User: {self.user.username} - Session ID: {self.session_id}"

def default_ear_list():
    return [-1, -1, -1]  # Return the list of default values

class SimpleEyeMetrics(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)  # Track the logged-in user
    session_id = models.IntegerField(default=0)  # Track login session
    video_id = models.IntegerField(default=0)  # Track video session
    timestamp = models.DateTimeField()  # Store the timestamp for each frame
    blink_count = models.IntegerField(null=True, blank=True)  # Store the blink count
    eye_aspect_ratio = models.FloatField(null=True, blank=True)  # Store the eye aspect ratio
    eyes_closed = models.BooleanField(default=False) # store if eyes are open in current frame
    x_coordinate_px = models.FloatField(null=True, blank=True)
    y_coordinate_px = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"User: {self.user.username} - Session: {self.session_id} - Video: {self.video_id} - Timestamp: {self.timestamp}"