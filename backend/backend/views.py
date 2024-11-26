from rest_framework_simplejwt.views import TokenObtainPairView
from django.contrib.auth.signals import user_logged_in
from django.contrib.auth.models import User

class CustomTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)  # Generate the token response
        try:
            # Fetch the user based on the username in the request
            username = request.data.get('username')
            user = User.objects.get(username=username)

            # Send the `user_logged_in` signal manually
            user_logged_in.send(sender=user.__class__, request=request, user=user)
            print(f"user_logged_in signal manually sent for: {user.username}")
        except User.DoesNotExist:
            print("User not found during login signal")
        except Exception as e:
            print(f"Error sending user_logged_in signal: {e}")

        return response
