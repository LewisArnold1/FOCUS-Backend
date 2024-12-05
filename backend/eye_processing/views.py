from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import SimpleEyeMetrics
from django.db.models import Max

class RetrieveLastBlinkCountView(APIView):

    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated
    #
    def get(self, request, *args, **kwargs):

        # Filter by user to retrieve the latest session ID and video ID
        current_session_id = SimpleEyeMetrics.objects.filter(user=request.user).aggregate(Max('session_id'))['session_id__max']
        latest_video_id = SimpleEyeMetrics.objects.filter(user=request.user, session_id=current_session_id).aggregate(Max('video_id'))['video_id__max']
        
        # Retrieve the last metric entry for the user, session, and video
        last_metric = SimpleEyeMetrics.objects.filter(user=request.user, session_id=current_session_id, video_id=latest_video_id).last()

        # Check if there is any data for this user
        if last_metric:
            data = {
                "blink_count": last_metric.blink_count
            }
        else:
            data = {
                "blink_count": 0  # Default value if no data exists for the user
            }
        # Send the blink count as a response
        return Response(data, status=200)

