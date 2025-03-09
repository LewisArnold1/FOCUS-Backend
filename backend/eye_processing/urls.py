from django.urls import path
from .views import RetrieveBlinkRateView, RetrieveReadingMetricsView, RetrieveBreakCheckView, RetrieveReadingSpeedView, RetrieveEyeMovementMetricsView

urlpatterns = [
    path('blink-rate/', RetrieveBlinkRateView.as_view(), name='blink-rate'),
    path('reading-times/', RetrieveReadingMetricsView.as_view(), name='reading-times'),
    path('break-check/', RetrieveBreakCheckView.as_view(), name='break-check'),
    path('reading-speed/', RetrieveReadingSpeedView.as_view(), name='reading-speed'),
    path('fix-sacc/', RetrieveEyeMovementMetricsView.as_view(), name='fix-sacc'),
]
