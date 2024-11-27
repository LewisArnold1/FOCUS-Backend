from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework import generics
from .serializers import RegisterUserSerializer
from django.contrib.auth.models import User

class RegisterUserView(generics.CreateAPIView):
    queryset = User.objects.all() # Ensure user does not already exist
    serializer_class = RegisterUserSerializer # Required data to create user
    serializer_class = RegisterUserSerializer # Required data to create user
    permission_classes = [AllowAny] # Any user should be able to register

class UserProfileView(generics.RetrieveAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = UserDisplaySerializer
    
    def get_object(self):
        self.request.session['init'] = True  # Write to session to generate a key
        return self.request.user
