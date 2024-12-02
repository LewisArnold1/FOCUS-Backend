from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import SimpleEyeMetrics
from django.db.models import Max

class RetrieveLastBlinkCountView(APIView):

    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated
    #
    def get(self, request, *args, **kwargs):

        # current session ID is largest value
        current_session_id = SimpleEyeMetrics.objects.filter(user=request.user).aggregate(Max('SessionId'))['SessionId__max']
        # find max video ID for this user & session ID
        max_video_id = SimpleEyeMetrics.objects.filter(user=request.user,SessionId=current_session_id).aggregate(Max('VideoID'))['VideoID__max']
        #max_video_id = this_session_metrics.aggregate(Max('VideoID'))['VideoID__max']
        # Filter by user, session & video
        last_metric = SimpleEyeMetrics.objects.filter(user=request.user,SessionId=current_session_id,video_id=max_video_id)
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

