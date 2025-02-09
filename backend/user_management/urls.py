from django.urls import path
from .views import RegisterUserView, ProfileView, CalibrationView, CalibrationRetrievalView, DocumentSaveView, DocumentLoadView, FileListView, FileDeleteView

# User-specific api end-points, so django routes request to appropriate user management views
urlpatterns = [
    path('register/', RegisterUserView.as_view(), name='register'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('calibrate/', CalibrationView.as_view(), name='calibrate'),
    path('calibration-retrieval/', CalibrationRetrievalView.as_view(), name='calibration-retrieval'),
    path('document-save', DocumentSaveView.as_view(), name='document-save'),
    path('document-load', DocumentLoadView.as_view(), name='document-load'),
    path('file-list/', FileListView.as_view(), name='file-list'),
    path('file-delete/', FileDeleteView.as_view(), name='file-delete')
]
