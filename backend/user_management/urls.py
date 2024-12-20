from django.urls import path
from .views import RegisterUserView, ProfileView, CalibrationView, CalibrationRetrievalView

# User-specific api end-points so when a user visits register or profile, django routes request to appropriate user management views
urlpatterns = [
    path('register/', RegisterUserView.as_view(), name='register'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('calibrate/', CalibrationView.as_view(), name='calibrate'),
    path('calibration-retrieval/', CalibrationRetrievalView.as_view(), name='calibration-retrieval'),
]
