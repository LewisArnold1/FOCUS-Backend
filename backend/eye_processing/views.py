from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import SimpleEyeMetrics, UserSession
from django.db.models import Max, Sum, Min
from datetime import timedelta
from sklearn.linear_model import LinearRegression  # Using linear regression for prediction
from eye_processing.eye_metrics.predict_blink_count import predict_blink_count

class RetrieveLastBlinkCountView(APIView):

    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated
    
    def get(self, request, *args, **kwargs):

        # Filter by user to retrieve the latest session ID and video ID
        current_session_id = SimpleEyeMetrics.objects.filter(user=request.user).aggregate(Max('session_id'))['session_id__max']
        latest_video_id = SimpleEyeMetrics.objects.filter(user=request.user, session_id=current_session_id).aggregate(Max('video_id'))['video_id__max']
        
        predicted_blink_count = predict_blink_count(request.user)
        
         # Retrieve all blink counts for this session and video to use for prediction
        metrics = SimpleEyeMetrics.objects.filter(
            user=request.user,
            session_id=current_session_id,
            video_id=latest_video_id
        ).order_by('timestamp')  # Ensure data is sorted by timestamp
        
        if not metrics.exists():
            return Response({"error": "No blink data available for prediction."}, status=404)
       
        total_blinks = metrics.aggregate(Sum('blink_count'))['blink_count__sum'] or 0
        blink_rates = self.calculate_blink_rate_per_minute(metrics)

        # Return the latest blink data and predicted blink count
        return Response({
            "predicted_blink_count": predicted_blink_count,
            "blink_rate_per_minute": blink_rates,
            "metrics": [
                {
                    "timestamp": metric.timestamp,
                    "blink_count": metric.blink_count,
                    "eye_aspect_ratio": metric.eye_aspect_ratio,
                    "x_coordinate": metric.x_coordinate_px,
                    "y_coordinate": metric.y_coordinate_px
                }
                for metric in metrics
            ]
        }, status=200)
        
    def calculate_blink_rate_per_minute(self, metrics):
        total_blinks = metrics.aggregate(Sum('blink_count'))['blink_count__sum'] or 0
        start_time = metrics.aggregate(Min('timestamp'))['timestamp__min']
        end_time = metrics.aggregate(Max('timestamp'))['timestamp__max']

        if start_time and end_time:
            duration_seconds = (end_time - start_time).total_seconds()
            if duration_seconds > 0:
                return [round((total_blinks * 60) / duration_seconds, 2)] 
        
        return [0]  

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
    
