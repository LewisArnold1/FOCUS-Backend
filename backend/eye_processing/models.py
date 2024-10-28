from django.db import models
from django.contrib.auth.models import User

class EyeMetrics(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100)
    frame_id = models.IntegerField(default=0)
    blink_count = models.IntegerField()
    eye_aspect_ratio = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"User: {self.user.username} - Session: {self.session_id} - Frame: {self.frame_id or 'N/A'}"