from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
#from eye_.models import SimpleEyeMetrics
from user_management.serializers import UserDisplaySerializer
from django.contrib.auth.models import User # remove?

class LoginView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        # Call the parent method to get the token response
        response = super().post(request, *args, **kwargs)
        return response
    def increment_login_id(self, request):
        
        # Get the user object from the request
        user = request.user
        print('test')
        # Check if the user is authenticated before printing or accessing its attributes
        if user.is_authenticated:
            print(f"Authenticated user: {user.username}")  # Should now correctly print the username
        else:
            print("User is not authenticated")
        '''
        # Increment the login_id for the user (you can add logic here if needed)
        if user.is_authenticated:
            # Find or create SimpleEyeMetrics entry for the user
            simple_eye_metrics, created = SimpleEyeMetrics.objects.get_or_create(user=user)
            # Increment the login_id
            simple_eye_metrics.login_id = simple_eye_metrics.login_id + 1 if simple_eye_metrics.login_id else 1
            simple_eye_metrics.save()
            '''
        #return response