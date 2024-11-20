from django.urls import path
from .views import RetrieveLastBlinkCountView

urlpatterns = [
    path('api/last-blink-count/', RetrieveLastBlinkCountView.as_view(), name='last-blink-count'),
]