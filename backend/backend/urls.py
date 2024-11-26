from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import (TokenRefreshView)
from .views import LoginView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/token/', LoginView.as_view(), name='login'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api-auth/', include("rest_framework.urls")),
    path('api/user/', include('user_management.urls')),
    path('api/eye/', include('eye_processing.urls')),
    path('api/session/', include('session_data.urls')),
]