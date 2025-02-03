from django.db import models
from django.contrib.auth.models import User

class UserSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)  # Track the logged-in user
    session_id = models.IntegerField(default=0)  # Track login session

    def __str__(self):
        return f"User: {self.user.username} - Session ID: {self.session_id}"


class SimpleEyeMetrics(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)  # Track the logged-in user
    session_id = models.IntegerField(default=0)  # Track login session
    video_id = models.IntegerField(default=0)  # Track video session
    timestamp = models.DateTimeField()  # Store the timestamp for each frame
    x_coordinate_px = models.FloatField(null=True, blank=True)
    y_coordinate_px = models.FloatField(null=True, blank=True)
    no_faces = models.IntegerField(null=True, blank=True)
    normalised_face_speed = models.FloatField(null=True, blank=True)
    eye_aspect_ratio = models.FloatField(null=True, blank=True)  # Store the eye aspect ratio
    blink_count = models.IntegerField(null=True, blank=True)  # Store the blink count
    left_centre = models.JSONField(null=True, blank=True)
    right_centre = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"User: {self.user.username} - Session: {self.session_id} - Video: {self.video_id} - Timestamp: {self.timestamp}"