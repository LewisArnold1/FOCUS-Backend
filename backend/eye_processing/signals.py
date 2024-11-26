from django.contrib.auth.signals import user_logged_in
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils.timezone import now
from .models import SimpleEyeMetrics

@receiver(user_logged_in)
def increment_login_id(sender, request, user, **kwargs):
    """
    This function increments the login_id for the user each time they log in.
    It's connected to the `user_logged_in` signal.
    """
    '''
    # Find or create the SimpleEyeMetrics entry for the user
    simple_eye_metrics, created = SimpleEyeMetrics.objects.get_or_create(user=user)
    
    # Increment the login_id
    simple_eye_metrics.login_id = simple_eye_metrics.login_id + 1 if simple_eye_metrics.login_id else 1
    simple_eye_metrics.save()

    print(f"User {user.username} logged in. Login ID incremented to {simple_eye_metrics.login_id}")
    '''
    print('test')
    #print(user.username)