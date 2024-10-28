from django.urls import path
from .views import RegisterUserView, UserProfileView

# User-specific api end-points so when a user visits register or profile, django routes request to appropriate user management views
urlpatterns = [
    path('register/', RegisterUserView.as_view(), name='register'),
    path('profile/', UserProfileView.as_view(), name='profile'),
]
