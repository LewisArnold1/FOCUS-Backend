from rest_framework import serializers
from .models import SimpleEyeMetrics

class VideoDataSerializer(serializers.Serializer):
    video_frame = serializers.ImageField()

    class Meta:
        fields = ['video_frame']

class SimpleEyeMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimpleEyeMetrics
        fields = ['timestamp', 'blink_count', 'eye_aspect_ratio'] # needs to change
