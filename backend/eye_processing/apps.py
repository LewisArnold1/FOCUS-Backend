from django.apps import AppConfig


class EyeProcessingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'eye_processing'

    #signals.py file used for login_id increment after user auth
    def ready(self): 
        import eye_processing.signals
