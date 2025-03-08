import numpy as np
from datetime import timedelta

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.db.models import Max, Min
from django.utils.timezone import now

from .models import SimpleEyeMetrics, UserSession

class RetrieveBlinkRateView(APIView):

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

        user_sessions = UserSession.objects.filter(user=user)

        if not user_sessions.exists():
            return Response({"error": "No sessions found for this user."}, status=404)

        session_blink_rates = []
        for session in user_sessions:
            session_rate, total_time = self.get_session_level_metrics(user, session.session_id, weighted=True)
            if session_rate is not None:
                session_blink_rates.append((session_rate, total_time))

        # Compute weighted average blink rate
        total_time_weighted = sum(time for _, time in session_blink_rates)
        if session_blink_rates and total_time_weighted > 0:
            avg_blink_rate = sum(rate * (time / total_time_weighted) for rate, time in session_blink_rates)
            return Response({"avg_blink_rate_per_session": round(avg_blink_rate, 2)}, status=200)

        return Response({"avg_blink_rate_per_session": None}, status=200)
    
    def get_session_level_metrics(self, user, session_id, weighted=False):
        
        video_data = SimpleEyeMetrics.objects.filter(user=user, session_id=session_id).values('video_id').distinct()

        video_blink_rates = []
        total_session_time = 0

        for video in video_data:
            video_id = video["video_id"]
            video_rate, video_time = self.get_video_level_metrics(user, session_id, video_id, weighted=True)
            if video_rate is not None:
                video_blink_rates.append((video_rate, video_time))
                total_session_time += video_time

        # Compute weighted avg for session if needed
        if video_blink_rates and total_session_time > 0:
            avg_blink_rate = sum(rate * (time / total_session_time) for rate, time in video_blink_rates)
            if weighted:
                return avg_blink_rate, total_session_time  # Used for user-level aggregation
            return None, 0

        return None, 0
    
    def get_video_level_metrics(self, user, session_id, video_id, weighted=False):
        blink_records = SimpleEyeMetrics.objects.filter(
            user=user, session_id=session_id, video_id=video_id
        ).order_by('timestamp')

        if not blink_records.exists():
            return (None, 0) if weighted else Response({"blink_rate_over_time": []}, status=200)

        blink_rate_over_time, total_video_time = self.calculate_blink_rate(blink_records)

        if weighted:
            avg_blink_rate = np.mean([rate["blink_rate"] for rate in blink_rate_over_time]) if blink_rate_over_time else None
            return avg_blink_rate, total_video_time

        # Downsample if necessary
        blink_rate_over_time = self.downsample_data(blink_rate_over_time)

        return Response({"blink_rate_over_time": blink_rate_over_time}, status=200)
    
    def calculate_blink_rate(self, blink_records):
        blink_values = [entry.blink_detected for entry in blink_records]
        blink_rate_over_time = []
        total_time = 0  # Total video time in minutes
        start_time = blink_records[0].timestamp
        current_time = start_time
        minute_blink_count = 0

        # Count blinks only when there are at least 3 consecutive ones
        i = 0
        while i < len(blink_values) - 2:
            if blink_values[i] == 1 and blink_values[i + 1] == 1 and blink_values[i + 2] == 1:
                minute_blink_count += 1
                # Skip to the end of this blink sequence to prevent overcounting
                while i < len(blink_values) and blink_values[i] == 1:
                    i += 1
            else:
                i += 1

            # Check if the minute has passed
            if (i - 1) >= 0 and (i - 1) < len(blink_records) and blink_records[i - 1].timestamp >= current_time + timedelta(minutes=1):
                # Store the blink rate for the last minute
                blink_rate_over_time.append({
                    "timestamp": current_time.isoformat(),
                    "blink_rate": minute_blink_count
                })
                total_time += 1  # Increment total video time
                current_time += timedelta(minutes=1)
                minute_blink_count = 0  # Reset for the new minute

        # Ensure last interval is recorded
        if minute_blink_count > 0:
            blink_rate_over_time.append({
                "timestamp": current_time.isoformat(),
                "blink_rate": minute_blink_count
            })
            total_time += 1

        return blink_rate_over_time, total_time 
    
    def downsample_data(self, data):
        if len(data) > 50:
            indices = np.linspace(0, len(data) - 1, 50).astype(int)
            return [data[i] for i in indices]
        return data

    
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
                "cumulative_time": total_time
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
                user=request.user, session_id=current_session_id, reading_mode__in=[2, 3, 4]
            ).aggregate(Max('video_id'))['video_id__max']

            if current_video_id is None:
                return Response({"error": "No valid reading videos found."}, status=400)

            return self.get_video_level_metrics(request.user, current_session_id, current_video_id)
        else:
            return Response({"error": "Invalid display parameter."}, status=400)

    def get_user_level_metrics(self, user):
        user_sessions = UserSession.objects.filter(user=user)

        if not user_sessions.exists():
            return Response({"error": "No sessions found for this user."}, status=404)

        session_reading_speeds = []
        for session in user_sessions:
            session_metrics = self.get_session_level_metrics(user, session.session_id, weighted=True)
            if session_metrics:
                session_reading_speeds.append(session_metrics)

        if not session_reading_speeds:
            return Response({"error": "No valid reading sessions found."}, status=404)

        total_words_read = sum(session["total_words_read"] for session in session_reading_speeds)
        total_time_weighted = sum(session["total_time"] for session in session_reading_speeds)

        avg_wpm = (sum(session["average_wpm"] * (session["total_time"] / total_time_weighted)
                       for session in session_reading_speeds) if total_time_weighted > 0 else None)

        response_data = {
            "average_wpm": round(avg_wpm, 2) if avg_wpm is not None else None,
            "total_words_read": round(total_words_read, 2),
            "sessions": session_reading_speeds
        }

        response_data["sessions"] = self.downsample_data(response_data["sessions"])
        return Response(response_data, status=200)

    def get_session_level_metrics(self, user, session_id, weighted=False):
        video_data = SimpleEyeMetrics.objects.filter(
            user=user, session_id=session_id, reading_mode__in=[2, 3, 4]
        ).values('video_id').distinct()

        if not video_data.exists():
            return None if weighted else Response({"error": "No valid reading videos found in this session."}, status=404)

        video_reading_speeds = []
        for video in video_data:
            video_metrics = self.get_video_level_metrics(user, session_id, video["video_id"], weighted=True)
            if video_metrics:
                video_reading_speeds.append(video_metrics)

        total_words_read = sum(video["total_words_read"] for video in video_reading_speeds)
        total_time_weighted = sum(video["total_time"] for video in video_reading_speeds)

        avg_wpm = (sum(video["average_wpm"] * (video["total_time"] / total_time_weighted)
                       for video in video_reading_speeds) if total_time_weighted > 0 else None)

        session_metrics = {
            "session_id": session_id,
            "average_wpm": round(avg_wpm, 2) if avg_wpm is not None else None,
            "total_words_read": round(total_words_read, 2),
            "total_time": total_time_weighted
        }

        if weighted:
            return session_metrics

        return Response({"videos": video_reading_speeds, **session_metrics}, status=200)

    def get_video_level_metrics(self, user, session_id, video_id, weighted=False):
        reading_records = SimpleEyeMetrics.objects.filter(
            user=user, session_id=session_id, video_id=video_id, reading_mode__in=[2, 3, 4]
        ).order_by("timestamp")

        if not reading_records.exists():
            return None if weighted else Response({
                "total_words_read": None,
                "average_wpm": None,
                "reading_speed_over_time": None
            }, status=200)

        total_words_read = 0
        reading_speed_over_time = []
        total_wpm = []
        prev_timestamp = None
        total_time = 0

        for record in reading_records:
            if record.wpm is not None:
                total_wpm.append(record.wpm)

                if prev_timestamp:
                    time_diff = (record.timestamp - prev_timestamp).total_seconds() / 60  # Convert to minutes
                    words_read = record.wpm * time_diff
                    total_words_read += words_read
                    total_time += time_diff

                    reading_speed_over_time.append({
                        "timestamp": record.timestamp.isoformat(),
                        "wpm": record.wpm
                    })

                prev_timestamp = record.timestamp

        avg_wpm = np.mean(total_wpm) if total_wpm else None

        video_metrics = {
            "video_id": video_id,
            "average_wpm": round(avg_wpm, 2) if avg_wpm is not None else None,
            "total_words_read": round(total_words_read, 2),
            "total_time": total_time
        }

        if weighted:
            return video_metrics

        video_metrics["reading_speed_over_time"] = self.downsample_data(reading_speed_over_time)
        return Response(video_metrics, status=200)

    def downsample_data(self, data):
        if len(data) > 50:
            indices = np.linspace(0, len(data) - 1, 50).astype(int)
            return [data[i] for i in indices]
        return data
    
class RetrieveEyeMovementMetricsView(APIView):
    permission_classes = [IsAuthenticated]  

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
        user_sessions = UserSession.objects.filter(user=user)

        if not user_sessions.exists():
            return Response({"error": "No sessions found for this user."}, status=404)

        session_movement_data = []
        for session in user_sessions:
            fixation_count, saccade_count, total_time = self.get_session_level_metrics(user, session.session_id, weighted=True)
            if fixation_count is not None:
                session_movement_data.append((fixation_count, saccade_count, total_time))

        total_time_weighted = sum(time for _, _, time in session_movement_data)

        if session_movement_data and total_time_weighted > 0:
            avg_fixation_count = sum(fix * (time / total_time_weighted) for fix, _, time in session_movement_data)
            avg_saccade_count = sum(sac * (time / total_time_weighted) for _, sac, time in session_movement_data)

            return Response({
                "avg_fixation_count_per_session": round(avg_fixation_count, 2),
                "avg_saccade_count_per_session": round(avg_saccade_count, 2),
            }, status=200)

        return Response({
            "avg_fixation_count_per_session": None,
            "avg_saccade_count_per_session": None,
        }, status=200)

    def get_session_level_metrics(self, user, session_id, weighted=False):
        video_data = SimpleEyeMetrics.objects.filter(user=user, session_id=session_id).values('video_id').distinct()

        video_movement_data = []
        total_session_time = 0

        for video in video_data:
            video_id = video["video_id"]
            fixation_count, saccade_count, video_time = self.get_video_level_metrics(user, session_id, video_id, weighted=True)
            if fixation_count is not None:
                video_movement_data.append((fixation_count, saccade_count, video_time))
                total_session_time += video_time

        if video_movement_data and total_session_time > 0:
            avg_fixation_count = sum(fix * (time / total_session_time) for fix, _, time in video_movement_data)
            avg_saccade_count = sum(sac * (time / total_session_time) for _, sac, time in video_movement_data)

            if weighted:
                return avg_fixation_count, avg_saccade_count, total_session_time

            return Response({
                "session_id": session_id,
                "fixation_count": round(avg_fixation_count, 2),
                "saccade_count": round(avg_saccade_count, 2),
            }, status=200)

        return None, None, 0

    def get_video_level_metrics(self, user, session_id, video_id, weighted=False):
        movement_records = SimpleEyeMetrics.objects.filter(
            user=user, session_id=session_id, video_id=video_id
        ).order_by("timestamp")

        if not movement_records.exists():
            return (None, None, 0) if weighted else Response({
                "fixation_count": 0,
                "saccade_count": 0
            }, status=200)

        fixation_count, saccade_count, total_time = self.calculate_fixation_saccade_count(movement_records)

        if weighted:
            return fixation_count, saccade_count, total_time

        return Response({
            "video_id": video_id,
            "fixation_count": fixation_count,
            "saccade_count": saccade_count,
        }, status=200)

    def calculate_fixation_saccade_count(self, movement_records, tolerance=1):
        fixation_count = 0
        saccade_count = 0
        in_fixation = False
        saccade_tolerance = 0  # Counter for tolerated saccades
        total_time = 0  # Total time in minutes
        prev_timestamp = None

        for record in movement_records:
            movement_type = record.movement_type  # Assuming 'fixation' or 'saccade'

            if prev_timestamp:
                time_diff = (record.timestamp - prev_timestamp).total_seconds() / 60  # Convert to minutes
                total_time += time_diff

            if movement_type == 'fixation':
                if not in_fixation:
                    fixation_count += 1  # Start new fixation
                    in_fixation = True
                saccade_tolerance = 0  # Reset tolerance counter

            elif movement_type == 'saccade':
                if in_fixation and saccade_tolerance < tolerance:
                    saccade_tolerance += 1  # Allow a tolerated saccade within fixation
                else:
                    saccade_count += 1  # Count independent saccades
                    in_fixation = False  # End fixation if tolerance is exceeded

            prev_timestamp = record.timestamp

        return fixation_count, saccade_count, total_time
