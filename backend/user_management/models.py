from django.db import models
from django.contrib.auth.models import User

class CalibrationData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    accuracy = models.IntegerField(null=True, blank=True)
    calibration_values = models.JSONField()

    def __str__(self):
        return f"CalibrationData for {self.user} at {self.created_at}, with an accuracy of {self.accuracy}"
