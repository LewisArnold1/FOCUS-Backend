# Session Data App Notes

## Overview:
This app handles tracking and storage of session-specific data such as blink count, eye aspect ratio, and session times. Each session is associated with a user and stores important metrics collected during a session.

### Key Endpoints:
- **Start Session**: `/api/session/start/` - Begins a new session and stores initial data.
- **Session History**: `/api/session/history/` - Retrieves the session history for the logged-in user.

## File Explanations:

### 1. `serializers.py`:
- **Purpose**: Serializes session data to JSON format for the API. It validates the data and converts it into a format suitable for storage in the database.
  
### 2. `views.py`:
- **Purpose**: Contains the logic for starting new sessions and retrieving session history. It ensures that the session is correctly tied to the logged-in user and saves metrics for each session.

### 3. `urls.py`:
- **Purpose**: Defines the routes (endpoints) for session-related actions, such as starting a session and retrieving session history.

### 4. `models.py`:
- **Purpose**: Defines the database schema for storing session data. Each session stores metrics like blink count and eye aspect ratio and is linked to a specific user.


## Example Workflow:
Start a Session:

A POST request is made to /api/session/start/ when the user starts a new session. A new SessionData entry is created, and the session begins tracking metrics (e.g., blinks, eye aspect ratio).
During the Session:

The backend continuously processes the video feed to extract eye metrics. These metrics (e.g., blinks, EAR) are stored temporarily and can be updated as the session progresses.
End the Session:

Once the session is complete, a POST request (or PUT if using updates) is sent to save the final metrics (blink count, EAR, etc.).
Retrieve Session History:

The user can view their session history by sending a GET request to /api/session/history/. The response will include a list of past sessions, along with the associated metrics.