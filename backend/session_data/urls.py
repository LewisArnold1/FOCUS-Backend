from django.urls import path
from .views import CreateSessionView, RetrieveSessionDataView, EndSessionView

urlpatterns = [
    path('start/', CreateSessionView.as_view(), name='start_session'),
    path('history/', RetrieveSessionDataView.as_view(), name='retrieve_session_data'),
    path('end/', EndSessionView.as_view(), name='end_session'), 
]