from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import RegisterUserSerializer
from django.contrib.auth.models import User
from .models import CalibrationData
from datetime import datetime

class RegisterUserView(generics.CreateAPIView):
    queryset = User.objects.all() # Ensure user does not already exist
    serializer_class = RegisterUserSerializer # Required data to create user
    permission_classes = [AllowAny] # Any user should be able to register

class CalibrationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # Get the authenticated user
        self.user = request.user

        # Extract and validate data from the request
        try:
            self.calibration_data, self.timestamp, self.accuracy = self.extract_data(request)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        try:
            calibration_entry = self.save_or_update_calibration()
            return Response(
                {"message": "Calibration data saved successfully.", "id": calibration_entry.id},
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            return Response(
                {"error": f"Failed to save calibration data: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
    
    def extract_data(self, request):

        # Extract data from the request
        calibration_data = request.data.get('data')
        timestamp = request.data.get('timestamp')
        accuracy = request.data.get('accuracy')

        if calibration_data is None or timestamp is None or accuracy is None:
            raise ValueError("Missing required fields: 'data', 'timestamp' and 'accuracy' are required.")

        if not isinstance(timestamp, (int, float)):
            raise ValueError("Invalid timestamp: must be a numeric value.")

        if not isinstance(accuracy, (int, float)):
            raise ValueError("Invalid accuracy: must be a numeric value.")

        # Convert timestamp
        timestamp_s = timestamp / 1000
        timestamp_dt = datetime.fromtimestamp(timestamp_s)
        
        return calibration_data, timestamp_dt, accuracy
    
    def save_or_update_calibration(self):

        calibration_entry, _ = CalibrationData.objects.update_or_create(
            user=self.user,
            defaults={
                'created_at': self.timestamp,
                'accuracy': self.accuracy,
                'calibration_values': self.calibration_data
            }
        )
        return calibration_entry
        

class CalibrationRetrievalView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated

    def get(self, request, *args, **kwargs):
        # Filter by the logged-in user and get the last entry
        calibration_data = CalibrationData.objects.filter(user=request.user)

        # Check if no data exists for this user
        if not calibration_data:
            return Response(
                {"error": "No calibration data found for this user."}, status=404
            )

        # If data exists, return it
        return Response(
            {
                "calibration_values": calibration_data.calibration_values,
                "created_at": calibration_data.created_at,
                "accuracy": calibration_data.accuracy
            }, 
            status=200
        )