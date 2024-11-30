from django.db import models
from django.contrib.auth.models import User

class CalibrationData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    calibration_values = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CalibrationData for {self.user} at {self.created_at}"
