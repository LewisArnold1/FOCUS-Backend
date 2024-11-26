from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from eye_processing.models import UserSession
from user_management.serializers import UserDisplaySerializer
from django.db.models import Max
from django.contrib.auth.models import User # remove?
from datetime import datetime

class LoginView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        # Get the serializer for token authentication
        serializer = self.get_serializer(data=request.data)
        
        # Ensure the serializer is valid (this happens automatically in the parent method too)
        if serializer.is_valid():
            # After validation, access the user associated with the credentials
            #user =   # This is the authenticated user
            print(serializer.user)
            # Increment the login_id field in the SimpleEyeMetrics model
            #simple_eye_metrics, created = SimpleEyeMetrics.objects.get_or_create(user=user)
            #simple_eye_metrics.login_id = (simple_eye_metrics.login_id or 0) + 1
            #simple_eye_metrics.save()

            # You can also print or log the user's information
            #print(f"User {user.username} logged in. Incremented login_id to {simple_eye_metrics.login_id}")

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
            return response
        else:
            raise ValidationError("Invalid credentials")
    
    '''
    def post(self, request, *args, **kwargs):
        # Call the parent method to get the token response
        response = super().post(request, *args, **kwargs)

        print(User.objects.all)()
        #print(user.username)
        """
        # Ensure the user is authenticated
        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            # At this point, the user is authenticated
            user = serializer.user  # Get the authenticated user from the serializer
            print(user)
            
            # Increment the login_id field in the SimpleEyeMetrics model
            simple_eye_metrics, created = SimpleEyeMetrics.objects.get_or_create(user=user)
            simple_eye_metrics.login_id = (simple_eye_metrics.login_id or 0) + 1
            simple_eye_metrics.save()
            
        else:
            print('wrong details')
"""
        return response
        '''
    #def increment_login_id(self, request):

'''
        # Get the user object from the request
        user = request.user
        print('test')
        # Check if the user is authenticated before printing or accessing its attributes
        if user.is_authenticated:
            print(f"Authenticated user: {user.username}")  # Should now correctly print the username
        else:
            print("User is not authenticated")
        
        # Increment the login_id for the user (you can add logic here if needed)
        if user.is_authenticated:
            # Find or create SimpleEyeMetrics entry for the user
            simple_eye_metrics, created = SimpleEyeMetrics.objects.get_or_create(user=user)
            # Increment the login_id
            simple_eye_metrics.login_id = simple_eye_metrics.login_id + 1 if simple_eye_metrics.login_id else 1
            simple_eye_metrics.save()
            '''
        #return response