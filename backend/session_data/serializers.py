from rest_framework import serializers
from .models import SessionData

class SessionDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = SessionData
        fields = ['id', 'user', 'start_time', 'end_time', 'blink_count', 'eye_aspect_ratio_avg']
