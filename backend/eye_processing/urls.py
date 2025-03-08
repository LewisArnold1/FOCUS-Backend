from django.urls import path
from .views import RetrieveBlinkRateView, RetrieveReadingMetricsView, RetrieveBreakCheckView, RetrieveReadingSpeedView

urlpatterns = [
    path('last-blink-count/', RetrieveBlinkRateView.as_view(), name='last-blink-count'),
    path('reading-times/', RetrieveReadingMetricsView.as_view(), name='reading-times'),
    path('break-check/', RetrieveBreakCheckView.as_view(), name='break-check'),
    path('reading-speed/', RetrieveReadingSpeedView.as_view(), name='reading-speed'),
]
