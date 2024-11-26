from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from django.db.models import Max
from eye_processing.models import SimpleEyeMetrics

@receiver(user_logged_in)
def initialise_session_id(sender, request, user, **kwargs):
    print("user_logged_in signal triggered")
    try: 
        # Get the latest session_id for the user
        last_session = SimpleEyeMetrics.objects.filter(user=user).aggregate(Max('session_id'))
        new_session_id = (last_session['session_id__max'] or 0) + 1

        # Create a new session entry
        request.session['current_session_id'] = new_session_id
        request.session.save()
        print(f"Session ID {new_session_id} initialised for user {user.username}")
    except Exception as e:
        print(f"Error in initialise_session_id: {e}")