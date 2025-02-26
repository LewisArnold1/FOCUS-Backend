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
    gaze_x = models.JSONField(null=True, blank=True)
    gaze_y = models.JSONField(null=True, blank=True)
    face_detected = models.BooleanField(default=False)
    normalised_eye_speed = models.FloatField(null=True, blank=True)
    face_yaw = models.FloatField(null=True, blank=True) 
    face_roll = models.FloatField(null=True, blank=True)
    face_pitch = models.FloatField(null=True, blank=True) 
    eye_aspect_ratio = models.FloatField(null=True, blank=True)  # Store the eye aspect ratio
    blink_detected = models.IntegerField(null=True, blank=True)  # Store the blink count
    left_centre = models.JSONField(null=True, blank=True)
    right_centre = models.JSONField(null=True, blank=True)
    focus = models.BooleanField(default=False)
    left_iris_velocity=models.FloatField(null=True, blank=True)
    right_iris_velocity=models.FloatField(null=True, blank=True)
    movement_type=models.CharField(max_length=10, default="fixation")
    frame=models.TextField(null=True, blank=True)
    reading_mode=models.IntegerField(default=3)
    wpm=models.IntegerField(default=0)

    def __str__(self):
        return f"User: {self.user.username} - Session: {self.session_id} - Video: {self.video_id} - Timestamp: {self.timestamp}"