from rest_framework import serializers

class VideoDataSerializer(serializers.Serializer):
    video_frame = serializers.ImageField()

    class Meta:
        fields = ['video_frame']
