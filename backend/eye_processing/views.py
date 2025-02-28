import numpy as np
from datetime import timedelta

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.db.models import Max, Min
from django.utils.timezone import now

from .models import SimpleEyeMetrics, UserSession

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
            if blink == 1: 
                if not blink_in_progress:  
                    minute_blink_count += 1 
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
    
class RetrieveReadingMetricsView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated

    def get(self, request, *args, **kwargs):
        display = request.query_params.get("display", "user")  
        current_session_id = SimpleEyeMetrics.objects.filter(user=request.user).aggregate(Max('session_id'))['session_id__max']

        if not current_session_id:
            return Response({"error": "No session data found."}, status=400)

        if display == "user":
            return self.get_user_level_metrics(request.user)
        elif display == "session":
            return self.get_session_level_metrics(request.user, current_session_id)
        elif display == "video":
            current_video_id = SimpleEyeMetrics.objects.filter(
                user=request.user, session_id=current_session_id
                ).aggregate(Max('video_id'))['video_id__max']

            if current_video_id is None:
                return Response({"error": "No video data found."}, status=400)

            return self.get_video_level_metrics(request.user, current_session_id, current_video_id)
        else:
            return Response({"error": "Invalid display parameter."}, status=400)

    def get_user_level_metrics(self, user):
        # Retrieve all sessions for the authenticated user
        user_sessions = UserSession.objects.filter(user=user)

        if not user_sessions.exists():
            return Response({"error": "No sessions found for this user."}, status=404)

        # Prepare session data
        sessions_data = []
        for session in user_sessions:
            total_reading_time, total_focus_time = self.calculate_total_session_times(user, session.session_id)

            sessions_data.append({
                "session_id": session.session_id,
                "total_reading_time": total_reading_time.total_seconds() / 60,
                "total_focus_time": total_focus_time.total_seconds() / 60,
            })

        # Downsample to 50 points
        if len(sessions_data) > 50:
            indices = np.linspace(0, len(sessions_data) - 1, 50).astype(int)
            sessions_data = [sessions_data[i] for i in indices]

        return Response({"sessions": sessions_data}, status=200)


    def get_session_level_metrics(self, user, session_id):
        # Get reading times for each video in this session
        video_data = SimpleEyeMetrics.objects.filter(user=user, session_id=session_id).values('video_id').distinct()

        video_metrics = []
        for video in video_data:
            video_id = video["video_id"]
            reading_time = self.calculate_reading_time(user, session_id, video_id)
            focus_time = self.calculate_focus_time(user, session_id, video_id)

            video_metrics.append({
                "video_id": video_id,
                "total_reading_time": reading_time.total_seconds() / 60,
                "total_focus_time": focus_time.total_seconds() / 60,
            })

        return Response({"videos": video_metrics}, status=200)
    
    def get_video_level_metrics(self, user, session_id, video_id):
        reading_time = self.calculate_cumulative_time(user, session_id, video_id, "reading_time")
        focus_time = self.calculate_cumulative_time(user, session_id, video_id, "focus_time")

        return Response({
            "cumulative_reading_time": reading_time,
            "cumulative_focus_time": focus_time,
        }, status=200)

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

        return (end_time - start_time) if start_time and end_time else timedelta(0)
    
    def calculate_focus_time(self, user, session_id, video_id):
        reading_time = self.calculate_reading_time(user, session_id, video_id)

        total_records = SimpleEyeMetrics.objects.filter(
            user=user, session_id=session_id, video_id=video_id
        ).count()

        if total_records == 0:
            return timedelta(0)

        focus_records = SimpleEyeMetrics.objects.filter(
            user=user, session_id=session_id, video_id=video_id, focus=True
        ).count()

        focus_percentage = (focus_records / total_records) if total_records > 0 else 0 
    
        return reading_time * focus_percentage

    def calculate_total_session_times(self, user, session_id):
        video_ids = SimpleEyeMetrics.objects.filter(
            user=user, session_id=session_id
        ).values_list('video_id', flat=True).distinct()

        total_reading_time = timedelta(0)
        weighted_focus_time = timedelta(0)

        for video_id in video_ids:
            video_reading_time = self.calculate_reading_time(user, session_id, video_id)
            video_focus_time = self.calculate_focus_time(user, session_id, video_id)

            total_reading_time += video_reading_time
            weighted_focus_time += video_focus_time

        return total_reading_time, weighted_focus_time
    
    def calculate_cumulative_time(self, user, session_id, video_id, time_field):
        records = SimpleEyeMetrics.objects.filter(
            user=user, session_id=session_id, video_id=video_id
        ).order_by("timestamp")

        if not records:
            return []

        cumulative_time = []
        total_time = 0  # Total minutes
        prev_timestamp = None

        for record in records:
            if prev_timestamp:
                time_diff = (record.timestamp - prev_timestamp).total_seconds() / 60  # Convert to minutes

                if time_field == "reading_time":
                    total_time += time_diff  # Always accumulate time
                elif time_field == "focus_time" and record.focus:
                    total_time += time_diff  # Only accumulate if focused

            cumulative_time.append({
                "timestamp": record.timestamp.isoformat(),
                "cumulative_time": round(total_time, 2)
            })

            prev_timestamp = record.timestamp  # Update for next iteration

        return cumulative_time
    
class RetrieveBreakCheckView(APIView):

    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated
    
    def get(self, request, *args, **kwargs):

        time_limit = float(request.query_params.get('time_limit', 1))  # Default to 1 minute

        # Filter by user to retrieve the latest session ID and video ID
        current_session_id = SimpleEyeMetrics.objects.filter(user=request.user).aggregate(Max('session_id'))['session_id__max']
        latest_video_id = SimpleEyeMetrics.objects.filter(user=request.user, session_id=current_session_id).aggregate(Max('video_id'))['video_id__max']
        if current_session_id is None or latest_video_id is None:
            return Response({"error": "No session or video data found."}, status=400)

        # Define the time window
        time_window = now() - timedelta(minutes=time_limit)

        # Retrieve data for the last time window 
        records = SimpleEyeMetrics.objects.filter(
            user=request.user, 
            session_id=current_session_id, 
            video_id=latest_video_id, 
            timestamp__gte=time_window
        )

        total_records = records.count()

        # Check if we have enough data to determine focus and face detection levels in time window
        if total_records < (time_limit * 60):
            return Response({"status": f"insufficient_data, found {total_records} records, current time: {now()} ... latest frame: {SimpleEyeMetrics.objects.order_by('timestamp').last().timestamp}"}, status=200)

        # Calculate the percentage of True values for focus and face_detected
        focus_true_count = records.filter(focus=True).count()
        face_detected_true_count = records.filter(face_detected=True).count()

        focus_percentage = (focus_true_count / total_records) * 100
        face_detected_percentage = (face_detected_true_count / total_records) * 100

        # Determine if user has been sufficiently focused and their face detected
        focus_status = focus_percentage >= 80
        face_detected_status = face_detected_percentage >= 80

        data = {
            "focus_status": focus_status,  
            "face_detected_status": face_detected_status,
        }

        return Response(data, status=200)
    
class RetrieveReadingSpeedView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated

    def get(self, request, *args, **kwargs):
        # Get latest session and video ID
        current_session_id = SimpleEyeMetrics.objects.filter(user=request.user).aggregate(Max('session_id'))['session_id__max']
        latest_video_id = SimpleEyeMetrics.objects.filter(user=request.user, session_id=current_session_id).aggregate(Max('video_id'))['video_id__max']

        if current_session_id is None or latest_video_id is None:
            return Response({"error": "No session or video data found."}, status=400)

        # Fetch relevant reading records (only modes 2, 3, 4)
        reading_records = SimpleEyeMetrics.objects.filter(
            user=request.user,
            session_id=current_session_id,
            video_id=latest_video_id,
            reading_mode__in=[2, 3, 4]
        ).order_by("timestamp")

        if not reading_records.exists():
            return Response({
                "total_words_read": None,
                "average_wpm": None,
                "reading_speed_over_time": None
            }, status=200)

        # Calculate reading speed metrics
        reading_speed_metrics = self.calculate_reading_speed_metrics(reading_records)

        return Response(reading_speed_metrics, status=200)

    def calculate_reading_speed_metrics(self, reading_records):
        total_words_read = 0
        reading_speed_over_time = []
        total_wpm = []
        prev_timestamp = None

        for record in reading_records:
            if record.wpm is not None:
                total_wpm.append(record.wpm)

                if prev_timestamp:
                    time_diff = (record.timestamp - prev_timestamp).total_seconds() / 60  # Convert to minutes
                    words_read = record.wpm * time_diff
                    total_words_read += words_read

                    reading_speed_over_time.append({
                        "timestamp": record.timestamp.isoformat(),
                        "wpm": record.wpm
                    })

                prev_timestamp = record.timestamp

        avg_wpm = np.mean(total_wpm) if total_wpm else None

        return {
            "total_words_read": round(total_words_read, 2),
            "average_wpm": round(avg_wpm, 2) if avg_wpm is not None else None,
            "reading_speed_over_time": reading_speed_over_time
        }