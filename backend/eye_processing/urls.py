from django.urls import path
from .views import RetrieveLastBlinkRateView, RetrieveAllUserSessionsView, RetrieveBreakCheckView

urlpatterns = [
    path('last-blink-count/', RetrieveLastBlinkRateView.as_view(), name='last-blink-count'),
    path('reading-times/', RetrieveAllUserSessionsView.as_view(), name='reading-times'),
    path('break-check', RetrieveBreakCheckView.as_view(), name='break-check'),
]
