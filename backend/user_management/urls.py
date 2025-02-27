from django.urls import path
from .views import VerifyEmailView, PasswordResetRequestView, PasswordResetConfirmView, RegisterUserView, ProfileView, CalibrationView, CalibrationRetrievalView, DocumentFirstSaveView, DocumentLoadView, DocumentUpdateView, FileListView, FileDeleteView, OnboardingView, OnboardingRetrievalView

# User-specific api end-points, so django routes request to appropriate user management views
urlpatterns = [
    path('verify-email/', VerifyEmailView.as_view(), name='verify-email'),
    path("password-reset/", PasswordResetRequestView.as_view(), name="password-reset"),
    path("password-reset-confirm/", PasswordResetConfirmView.as_view(), name="password-reset-confirm"),
    path('register/', RegisterUserView.as_view(), name='register'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('calibrate/', CalibrationView.as_view(), name='calibrate'),
    path('calibration-retrieval/', CalibrationRetrievalView.as_view(), name='calibration-retrieval'),
    path('onboarding/', OnboardingView.as_view(), name='onboarding'),
    path('onboarding-retrieval/', OnboardingRetrievalView.as_view(), name='onboarding-retrieval'),
    path('document-save', DocumentFirstSaveView.as_view(), name='document-save'),
    path('document-load', DocumentLoadView.as_view(), name='document-load'),
    path('document-update', DocumentUpdateView.as_view(), name='document-update'), 
    path('file-list/', FileListView.as_view(), name='file-list'),
    path('file-delete', FileDeleteView.as_view(), name='file-delete')
]
