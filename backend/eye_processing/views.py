from rest_framework.views import APIView
from rest_framework.response import Response
from .models import EyeMetrics

class ProcessVideoFrameView(APIView):
    def post(self, request, *args, **kwargs):
        user = request.user  # Current logged-in user
        session_id = request.data.get('session_id')
        blink_count = request.data.get('blink_count')
        eye_aspect_ratio = request.data.get('eye_aspect_ratio')

        # Store the raw data in EyeMetrics
        EyeMetrics.objects.create(
            user=user,
            session_id=session_id,
            blink_count=blink_count,
            eye_aspect_ratio=eye_aspect_ratio,
        )
        return Response({"message": "Frame processed"}, status=200)

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
