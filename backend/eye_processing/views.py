from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import SimpleEyeMetrics, UserSession
from django.db.models import Max, Sum, Min
from datetime import timedelta

class RetrieveLastBlinkCountView(APIView):

    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated
    
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
            session_total_time = self.calculate_total_session_reading_time(
                request.user, session.session_id
            )

            # Get reading times for each video in this session
            video_reading_times = SimpleEyeMetrics.objects.filter(
                user=request.user, session_id=session.session_id
            ).values('video_id').distinct()

            # Format video reading times
            video_data = [
                {
                    "video_id": video["video_id"],
                    "total_reading_time": self.calculate_reading_time(
                        request.user, session.session_id, video["video_id"]
                    ),
                }
                for video in video_reading_times
            ]

            # Add session details to the response
            sessions_data.append({
                "session_id": session.session_id,
                "total_reading_time": session_total_time,
                "videos": video_data,
            })

        # Return all sessions data
        return Response({"sessions": sessions_data}, status=200)

    def calculate_reading_time(self, user, session_id, video_id):
        # Get the earliest and latest timestamps for the session and video
        timestamps = SimpleEyeMetrics.objects.filter(
            user=user,
            session_id=session_id,
            video_id=video_id
        ).aggregate(
            start_time=Min('timestamp'),
            end_time=Max('timestamp')
        )

        # Calculate the reading time
        start_time = timestamps['start_time']
        end_time = timestamps['end_time']

        if start_time and end_time:
            return end_time - start_time
        else:
            return timedelta(0)  # Return 0 if no data is available

    def calculate_total_session_reading_time(self, user, session_id):
        # Get unique video IDs for the session
        video_ids = SimpleEyeMetrics.objects.filter(
            user=user, session_id=session_id
        ).values_list('video_id', flat=True).distinct()

        # Calculate total reading time by summing reading times across all videos
        total_time = timedelta(0)
        for video_id in video_ids:
            total_time += self.calculate_reading_time(user, session_id, video_id)

        return total_time
    
