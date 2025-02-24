from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import SimpleEyeMetrics, UserSession
from django.db.models import Max, Sum, Min
from datetime import timedelta

class RetrieveLastBlinkRateView(APIView):

    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated
    
    def get(self, request, *args, **kwargs):

        # Filter by user to retrieve the latest session ID and video ID
        current_session_id = SimpleEyeMetrics.objects.filter(user=request.user).aggregate(Max('session_id'))['session_id__max']
        latest_video_id = SimpleEyeMetrics.objects.filter(user=request.user, session_id=current_session_id).aggregate(Max('video_id'))['video_id__max']
        
        # Retrieve blink timestamps for blink rate calculation
        blink_records = SimpleEyeMetrics.objects.filter(
            user=request.user, session_id=current_session_id, video_id=latest_video_id
        ).order_by('timestamp')

        blink_rate = self.calculate_blink_rate(blink_records)

        # Only send blink_rate
        data = {
            "blink_rate": blink_rate  # Return only blink rate per minute as an array
        }
        
        return Response(data, status=200)
    
    def calculate_blink_rate(self, blink_timestamps):
        """
        Calculate blink rate per minute from blink timestamps.
        Treats consecutive 1s as one blink until there is a 0.
        """
        if not blink_timestamps.exists():
            return []

        # Extract blink counts (0 or 1) per frame from the database
        blink_counts = [entry.blink_count for entry in blink_timestamps]
        
        # Initialize variables
        blink_rates = []  # Store the number of blinks per minute
        blink_in_progress = False  # Flag to track ongoing blink
        current_blink_count = 0  # Count of blinks in the current minute

        start_time = blink_timestamps[0].timestamp
        end_time = blink_timestamps[-1].timestamp

        current_time = start_time
        minute_blink_count = 0

        # Loop through each timestamp and calculate blink rate
        for index, blink in enumerate(blink_counts):
            if blink == 1:  # Blink detected
                if not blink_in_progress:  # If it's the start of a blink
                    minute_blink_count += 1  # Count a blink
                    blink_in_progress = True
            else:  # Blink ended (0 detected)
                blink_in_progress = False  # Reset the blink flag

            # Check if the minute has passed (based on timestamp)
            if blink_timestamps[index].timestamp >= current_time + timedelta(minutes=1):
                # Store the blink rate for the last minute
                blink_rates.append(minute_blink_count)
                # Move to the next minute
                current_time = current_time + timedelta(minutes=1)
                minute_blink_count = 0  # Reset for the new minute

        # Make sure the last minute is added if there are remaining blinks
        if minute_blink_count > 0:
            blink_rates.append(minute_blink_count)

        return blink_rates
    
    def calculate_blink_rate(self, blink_timestamps):
        """
        Calculate blink rate per minute from blink timestamps.
        """
        if not blink_timestamps.exists():
            return []

        timestamps = [entry.timestamp for entry in blink_timestamps]
        start_time = timestamps[0]
        end_time = timestamps[-1]
        blink_rates = []
        current_time = start_time

        while current_time <= end_time:
            next_minute = current_time + timedelta(minutes=1)
            blink_count = sum(1 for t in timestamps if current_time <= t < next_minute)
            blink_rates.append(blink_count)
            current_time = next_minute

        return blink_rates

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
    
