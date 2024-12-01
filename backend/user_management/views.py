from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import RegisterUserSerializer
from django.contrib.auth.models import User
from .models import CalibrationData

class RegisterUserView(generics.CreateAPIView):
    queryset = User.objects.all() # Ensure user does not already exist
    serializer_class = RegisterUserSerializer # Required data to create user
    permission_classes = [AllowAny] # Any user should be able to register

class CalibrationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # Get the authenticated user
        user = request.user
        
        # Extract data from the request
        calibration_data = request.data.get('data')
        timestamp = request.data.get('timestamp')

        # Validate the data (optional, depends on your requirements)
        if not calibration_data or not timestamp:
            return Response({"error": "Missing required fields."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Save the data to the database
            calibration_entry = CalibrationData.objects.create(
                user=user,
                calibration_values=calibration_data,  # Assuming this is a JSONField in your model
                created_at=timestamp  # Assuming your model has a DateTimeField
            )

            return Response(
                {"message": "Calibration data saved successfully.", "id": calibration_entry.id},
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            return Response({"error": f"Failed to save data: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class CalibrationRetrievalView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated

    def get(self, request, *args, **kwargs):
        # Filter by the logged-in user and get the last entry
        calibration_data = CalibrationData.objects.filter(user=request.user).last()

        # Check if no data exists for this user
        if not calibration_data:
            return Response(
                {"error": "No calibration data found for this user."}, status=404
            )

        # If data exists, return it
        return Response(
            {"calibration_values": calibration_data.calibration_values}, status=200
        )