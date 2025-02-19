from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.exceptions import ValidationError
from eye_processing.models import UserSession
from django.db.models import Max

class LoginView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        # Get the serializer for token authentication
        serializer = self.get_serializer(data=request.data)
        
        # Validate then access the user associated with the credentials
        if serializer.is_valid():
            # Get max login_id
            max_session_id = UserSession.objects.filter(user=serializer.user).aggregate(Max('session_id'))['session_id__max'] or 0
            new_session = UserSession(
                user = serializer.user,  # Associate the logged-in user
                session_id = max_session_id + 1 #Increment sesson ID
            )
            new_session.save()
            print(max_session_id + 1)
            # Call the parent method to generate and return the token
            response = super().post(request, *args, **kwargs)
            response.data['session_id'] = new_session.session_id
            
            return response
        else:
            raise ValidationError("Invalid credentials")