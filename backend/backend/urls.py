from django.contrib import admin
from django.urls import path, include
from backend.views import CustomTokenObtainPairView  # Import the custom view
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),  # Use custom view
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api-auth/', include("rest_framework.urls")),
    path('api/user/', include('user_management.urls')),
    path('api/eye/', include('eye_processing.urls')),
]
