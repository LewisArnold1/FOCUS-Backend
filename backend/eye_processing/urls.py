from django.urls import path
from .views import RetrieveLastBlinkRateView, RetrieveAllUserSessionsView

urlpatterns = [
    path('last-blink-count/', RetrieveLastBlinkRateView.as_view(), name='last-blink-count'),
    path('reading-times/', RetrieveAllUserSessionsView.as_view(), name='reading-times'),
]
