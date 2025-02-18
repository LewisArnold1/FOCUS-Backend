from datetime import datetime
import mimetypes
import os
import base64

from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth.models import User
from django.http import FileResponse
from django.conf import settings
from django.shortcuts import get_object_or_404
import os

from .serializers import RegisterUserSerializer
from .models import CalibrationData, DocumentData, OnboardingData

class RegisterUserView(generics.CreateAPIView):
    queryset = User.objects.all() # Ensure user does not already exist
    serializer_class = RegisterUserSerializer # Required data to create user
    permission_classes = [AllowAny] # Any user should be able to register

class ProfileView(APIView):
   permission_classes = [IsAuthenticated]

   def get(self, request, *args, **kwargs):
        # Return the username of the authenticated user
        return Response({"username": request.user.username}, status=status.HTTP_200_OK)

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
        try:
            # Filter by the logged-in user and get the last entry
            calibration_data = CalibrationData.objects.get(user=request.user)

            # If data exists, return it
            return Response(
                {
                    "calibration_values": calibration_data.calibration_values,
                    "created_at": calibration_data.created_at,
                    "accuracy": calibration_data.accuracy
                },
                status=200
            )
        except CalibrationData.DoesNotExist:
            # If no data exists for this user
            return Response(
                {"error": "No calibration data found for this user."}, status=404
            )
        
class DocumentFirstSaveView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # Get the authenticated user
        self.user = request.user

        # Extract and validate data from the request
        try:
            self.file_name, self.file_object, self.line_number, self.page_number, self.timestamp = self.extract_data(request)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        try:
            document_entry = self.save_or_update_document_drive()
            return Response(
                {"message": "File progress data saved successfully."},
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            return Response(
                {"error": f"Failed to save file progress data: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        
    def extract_data(self, request):
        
        # Extract data from the request
        file_name = request.POST.get('file_name')
        file_object = request.FILES.get('file_object')
        line_number = int(request.POST.get('line_number'))
        page_number = int(request.POST.get('page_number'))
        timestamp = int(request.POST.get('timestamp'))

        required_fields = {
        "file_name": file_name,
        "file_object": file_object,
        "line_number": line_number,
        "page_number": page_number,
        "timestamp": timestamp,
        }

        missing_fields = [field for field, value in required_fields.items() if value is None]

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Validate data types
        if not isinstance(file_name, str):
            raise ValueError("Invalid file name: must be a string value.")

        if not isinstance(line_number, int):
            raise ValueError("Invalid line number: must be an integer.")

        if not isinstance(page_number, int):
            raise ValueError("Invalid page number: must be an integer.")

        if not isinstance(timestamp, (int, float)):
            raise ValueError("Invalid timestamp: must be a numeric value (int or float).")

        valid_mime_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']

        mime_type, _ = mimetypes.guess_type(file_object.name)
        if mime_type not in valid_mime_types:
            raise ValueError("Invalid file type. Only .pdf, .docx and .txt are allowed.")
       
        # Convert timestamp
        timestamp_s = timestamp / 1000
        timestamp_dt = datetime.fromtimestamp(timestamp_s)
        
        return file_name, file_object, line_number, page_number, timestamp_dt
    
    def save_or_update_document_drive(self):

        document_entry, _ = DocumentData.objects.update_or_create(
        user=self.user,
        file_name=self.file_name,
        defaults={
            'file_object': self.file_object,
            'line_number': self.line_number,
            'page_number': self.page_number,
            'saved_at': self.timestamp
            }
        )
        return document_entry

class DocumentUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # Get the authenticated user
        self.user = request.user

        # Extract and validate data from the request
        try:
            extracted_data = self.extract_data(request)
            self.file_name = extracted_data.get("file_name")
            self.new_file_name = extracted_data.get("new_file_name")
            self.line_number = extracted_data.get("line_number")
            self.page_number = extracted_data.get("page_number")
            self.timestamp = extracted_data.get("timestamp")  # Required field
            self.favourite = extracted_data.get("favourite")  
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        try:
            document_entry = self.update_document_metadata()
            return Response(
                {"message": "File progress data updated successfully."},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"error": f"Failed to update file progress data: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        
    def extract_data(self, request):
        file_name = request.POST.get('file_name') # Required
        new_file_name = request.POST.get('new_file_name') 
        line_number = request.POST.get('line_number')
        page_number = request.POST.get('page_number')
        timestamp = float(request.POST.get('timestamp')) # Required
        favourite = request.POST.get('favourite')

        # Ensure required fields are present
        if file_name is None or timestamp is None:
            raise ValueError("Missing required fields: 'file_name' and 'timestamp' are required.")

        # Validate data types (only if provided)
        if not isinstance(file_name, str):
            raise ValueError("Invalid file name: must be a string value.")
        
        if new_file_name is not None and not isinstance(new_file_name, str):
            raise ValueError("Invalid new file name: must be a string value.")

        if line_number is not None:
            line_number = int(line_number)
            if not isinstance(line_number, int) or line_number < 0:
                raise ValueError("Invalid line number: must be a non-negative integer.")

        if page_number is not None:
            page_number = int(page_number)
            if not isinstance(page_number, int) or page_number < 0:
                raise ValueError("Invalid page number: must be a non-negative integer.")
            
        if not isinstance(timestamp, (int, float)):
            raise ValueError("Invalid timestamp: must be a numeric value (int or float).")

        # Convert timestamp
        timestamp_s = timestamp / 1000
        timestamp_dt = datetime.fromtimestamp(timestamp_s)

        # Convert `favourite` to boolean (handles string "true"/"false")
        if favourite is not None:
            if(isinstance(favourite, str)):
                favourite = favourite.lower() == "true"
            if not isinstance(favourite, bool):
                raise ValueError("Invalid favourite value: must be a boolean (true/false).")
  
        return {
            "file_name": file_name,
            "new_file_name": new_file_name,
            "line_number": line_number,
            "page_number": page_number,
            "timestamp": timestamp_dt,
            "favourite": favourite,
        }
    
    def update_document_metadata(self):
        """ Updates metadata fields of an existing document """
        
        # Ensure the document exists before updating
        document_entry = get_object_or_404(DocumentData, user=self.user, file_name=self.file_name)

        # Update only the fields that were provided
        if self.new_file_name:
            document_entry.file_name = self.new_file_name
        if self.line_number is not None:
            document_entry.line_number = self.line_number
        if self.page_number is not None:
            document_entry.page_number = self.page_number
        if self.favourite is not None:
            document_entry.favourite = self.favourite

        # Always update timestamp
        document_entry.saved_at = self.timestamp
        document_entry.save()

        return document_entry

class DocumentLoadView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated

    def get(self, request, *args, **kwargs):
        try:
            from user_management.models import DocumentData
            # Retrieve the document data
            document_data = DocumentData.objects.get(user=request.user, file_name=request.query_params.get('file_name'))

            # Get the file path
            file_path = document_data.file_object.path

            # Ensure the file exists
            if not os.path.exists(file_path):
                return Response({"error": "File not found on the server."}, status=404)

            # Create the file response
            response = FileResponse(open(file_path, 'rb'), as_attachment=True, filename=document_data.file_name)

            # Add metadata to headers
            response["line-number"] = document_data.line_number
            response["page-number"] = document_data.page_number

            return response

        except DocumentData.DoesNotExist:
            return Response({"error": "No document data found for this user."}, status=404)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
        
class FileListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        try:
            documents = DocumentData.objects.filter(user=request.user)

            files = []
            for document in documents:
                # Generate preview if needed
                preview_path = document.generate_preview()
                # Read the preview file as Base64 (if it exists)
                preview_base64 = None
                if preview_path and os.path.exists(preview_path):
                    with open(preview_path, "rb") as preview_file:
                        preview_base64 = base64.b64encode(preview_file.read()).decode('utf-8')
                else:
                    print("Preview file not found:", preview_path)

                files.append({
                    'name': document.file_name,
                    'thumbnail': preview_base64,  # Base64 encoded image content
                    'isStarred': document.favourite,
                    'lastOpened': document.saved_at.timestamp() * 1000,
                })

            return Response(files, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)
        
class FileDeleteView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, *args, **kwargs):
        try:
            file_name = request.query_params.get('file_name')
            document = DocumentData.objects.get(user=request.user, file_name=file_name)

            preview_filename = f"{os.path.basename(os.path.splitext(document.file_object.name)[0])}_preview.jpg"
            preview_path = os.path.join(settings.MEDIA_ROOT, "documents", preview_filename)

            # print("Preview path:", preview_path)
            if os.path.exists(preview_path):
                os.remove(preview_path)

            # Delete the actual file
            document.file_object.delete()

            # Delete the database entry
            document.delete()
            return Response({"message": "File deleted successfully."}, status=200)

        except DocumentData.DoesNotExist:
            return Response({"error": "File not found."}, status=404)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
        
class OnboardingView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # Get the authenticated user
        self.user = request.user

        # Extract and validate data from the request
        try:
            self.name, self.dob, self.screen_time, self.sleep_time, self.eye_strain, self.glasses, self.timestamp = self.extract_data(request)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        try:
            onboarding_entry = self.save_or_update_onboarding()
            return Response(
                {"message": "Onboarding data saved successfully.", "id": onboarding_entry.id},
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            return Response(
                {"error": f"Failed to save onboarding data: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def extract_data(self, request):
        # Extract data from request
        name = request.data.get("name")
        dob = request.data.get("dob")  # Date of birth should be in YYYY-MM-DD format
        screen_time = request.data.get("screen_time")
        sleep_time = request.data.get("sleep_time")
        eye_strain = request.data.get("eye_strain")
        glasses = request.data.get("glasses")
        timestamp = request.data.get("timestamp")

        required_fields = {
            "name": name,
            "dob": dob,
            "screen_time": screen_time,
            "sleep_time": sleep_time,
            "eye_strain": eye_strain,
            "glasses": glasses,
            "timestamp": timestamp
        }

        # Check for missing fields
        missing_fields = [field for field, value in required_fields.items() if value is None]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Validate data types
        if not isinstance(name, str):
            raise ValueError("Invalid name: must be a string.")

        try:
            dob = datetime.strptime(dob, "%Y-%m-%d").date()  # Convert to date format
        except ValueError:
            raise ValueError("Invalid DOB format: must be YYYY-MM-DD.")

        if not isinstance(screen_time, int) or screen_time < 0:
            raise ValueError("Invalid screen time: must be a positive integer.")

        if not isinstance(sleep_time, int) or sleep_time < 0:
            raise ValueError("Invalid sleep time: must be a positive integer.")

        if not isinstance(eye_strain, bool):
            raise ValueError("Invalid eye strain value: must be a boolean.")

        if not isinstance(glasses, bool):
            raise ValueError("Invalid glasses value: must be a boolean.")

        if not isinstance(timestamp, (int, float)):
            raise ValueError("Invalid timestamp: must be a numeric value.")

        # Convert timestamp
        timestamp_s = timestamp / 1000
        timestamp_dt = datetime.fromtimestamp(timestamp_s)

        return name, dob, screen_time, sleep_time, eye_strain, glasses, timestamp_dt

    def save_or_update_onboarding(self):
        onboarding_entry, _ = OnboardingData.objects.update_or_create(
            user=self.user,
            defaults={
                "name": self.name,
                "dob": self.dob,
                "screen_time": self.screen_time,
                "sleep_time": self.sleep_time,
                "eye_strain": self.eye_strain,
                "glasses": self.glasses,
                "created_at": self.timestamp,
            }
        )
        return onboarding_entry


class OnboardingRetrievalView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure user is authenticated

    def get(self, request, *args, **kwargs):
        try:
            # Get the latest onboarding data for the user
            onboarding_data = OnboardingData.objects.get(user=request.user)

            return Response(
                {
                    "name": onboarding_data.name,
                    "dob": onboarding_data.dob.strftime("%Y-%m-%d") if onboarding_data.dob else None,  # Format DOB
                    "screen_time": onboarding_data.screen_time,
                    "sleep_time": onboarding_data.sleep_time,
                    "eye_strain": onboarding_data.eye_strain,
                    "glasses": onboarding_data.glasses,
                    "created_at": onboarding_data.created_at,
                },
                status=200
            )
        except OnboardingData.DoesNotExist:
            return Response({"error": "No onboarding data found for this user."}, status=404)