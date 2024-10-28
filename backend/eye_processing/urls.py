from django.urls import path
from .views import ProcessVideoFrameView, RetrieveEyeMetricsView

urlpatterns = [
    path('process/', ProcessVideoFrameView.as_view(), name='process_eye_data'),  # Handles frame processing
    path('metrics/', RetrieveEyeMetricsView.as_view(), name='retrieve_eye_metrics'),  # Retrieves metrics for a user
]