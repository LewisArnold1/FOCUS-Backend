from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import SimpleEyeMetrics, UserSession
from django.db.models import Max, Sum
from datetime import timedelta

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


class RetrieveAllUserSessionsView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated

    def get(self, request, *args, **kwargs):
        # Retrieve all sessions for the authenticated user
        user_sessions = UserSession.objects.filter(user=request.user)

        if not user_sessions.exists():
            return Response({"error": "No sessions found for this user."}, status=404)

        # Prepare session data
        sessions_data = []
        for session in user_sessions:
            # Get total reading time for the session
            session_total_time = SimpleEyeMetrics.objects.filter(
                user=request.user, session_id=session.session_id
            ).aggregate(total_time=Sum('reading_time'))['total_time'] or timedelta(0)

            # Get reading times for each video in this session
            video_reading_times = SimpleEyeMetrics.objects.filter(
                user=request.user, session_id=session.session_id
            ).values('video_id').annotate(total_time=Sum('reading_time'))

            # Format video reading times
            video_data = [
                {"video_id": video["video_id"], "total_reading_time": str(video["total_time"])}
                for video in video_reading_times
            ]

            # Add session details to the response
            sessions_data.append({
                "session_id": session.session_id,
                "total_reading_time": str(session_total_time),
                "videos": video_data,
                "date": session.session_id,  # Assuming `session_id` encodes session creation. Replace with proper field if necessary.
            })

        # Return all sessions data
        return Response({"sessions": sessions_data}, status=200)
    


