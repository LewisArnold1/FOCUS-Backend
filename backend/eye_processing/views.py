from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import SimpleEyeMetrics

class RetrieveLastBlinkCountView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated
    def get(self, request, *args, **kwargs):
        
        # current session ID will always be most recent entry
        last_metric = SimpleEyeMetrics.objects.filter(user=request.user).last() # to be changed to max (with incrementing id)
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