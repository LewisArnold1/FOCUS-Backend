from django.urls import path
from .views import RetrieveLastBlinkCountView, RetrieveAllUserSessionsView

urlpatterns = [
    path('api/last-blink-count/', RetrieveLastBlinkCountView.as_view(), name='last-blink-count'),
    path('reading-times/', RetrieveAllUserSessionsView.as_view(), name='reading-times'),
]
