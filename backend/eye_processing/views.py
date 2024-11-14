from rest_framework.views import APIView
from rest_framework.response import Response
from .models import EyeMetrics

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
