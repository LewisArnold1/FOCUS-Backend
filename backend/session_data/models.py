from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class SessionData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100, default="default_session")
    total_blinks = models.IntegerField()
    avg_eye_aspect_ratio = models.FloatField()
    session_start = models.DateTimeField(default=timezone.now)
    session_end = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"User: {self.user.username} - Session: {self.session_id}"