from rest_framework import generics
from .serializers import SessionDataSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import Avg, Sum
from eye_processing.models import EyeMetrics 
from session_data.models import SessionData
from django.utils.timezone import now

class CreateSessionView(generics.CreateAPIView):
    queryset = SessionData.objects.all()
    serializer_class = SessionDataSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        # Automatically set the session's user to the currently authenticated user
        serializer.save(user=self.request.user)

class RetrieveSessionDataView(generics.ListAPIView):
    serializer_class = SessionDataSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Only return sessions that belong to the currently authenticated user
        return SessionData.objects.filter(user=self.request.user)

class EndSessionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # Get session ID from request
        session_id = request.data.get('session_id')

        # Ensure the session exists in EyeMetrics
        eye_metrics = EyeMetrics.objects.filter(user=request.user, session_id=session_id)

        if not eye_metrics.exists():
            return Response({"error": "No data found for this session"}, status=404)

        # Aggregate the blink count and average eye aspect ratio
        total_blinks = eye_metrics.aggregate(Sum('blink_count'))['blink_count__sum']
        avg_eye_aspect_ratio = eye_metrics.aggregate(Avg('eye_aspect_ratio'))['eye_aspect_ratio__avg']

        # Update or create session summary data in SessionData
        SessionData.objects.update_or_create(
            user=request.user,
            session_id=session_id,
            defaults={
                'total_blinks': total_blinks,
                'avg_eye_aspect_ratio': avg_eye_aspect_ratio,
                'session_end': now()  # End the session now
            }
        )

        return Response({"message": "Session ended and data aggregated"}, status=200)
