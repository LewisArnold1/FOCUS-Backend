from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import SimpleEyeMetrics

class RetrieveLastBlinkCountView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated

    def get(self, request, *args, **kwargs):
        # Filter by the logged-in user and get the last entry
        last_metric = SimpleEyeMetrics.objects.filter(user=request.user).last()

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
"""
class RetrieveEyeMetricsView(APIView):
    def get(self, request, *args, **kwargs):
        # Filter by session_id
        session_id = request.query_params.get('session_id', None)
        if session_id:
            metrics = EyeMetrics.objects.filter(user=request.user, session_id=session_id)
        else:
            metrics = EyeMetrics.objects.filter(user=request.user)

        # Prepare the response with metrics
        data = [
            {
                "session_id": metric.session_id,
                "blink_count": metric.blink_count,
                "eye_aspect_ratio": metric.eye_aspect_ratio,
                "frame_id": metric.frame_id,
            } for metric in metrics
        ]
        return Response(data, status=200)
"""
