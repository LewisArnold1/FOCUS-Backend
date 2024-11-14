from django.urls import path
from .views import RetrieveEyeMetricsView

urlpatterns = [
    path('metrics/', RetrieveEyeMetricsView.as_view(), name='retrieve_eye_metrics'),  # Retrieves metrics for a user
]