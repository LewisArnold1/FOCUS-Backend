# User Management App Notes

## Overview:
This app handles user registration, login, and profile management using Django Rest Framework and JWT for authentication.

### Key Endpoints:
- **Register User**: `/api/user/register/` - Accepts user data (username, email, password) and creates a new user.
- **User Profile**: `/api/user/profile/` - Retrieves the profile of the currently logged-in user.
- **Token Authentication**:
  - Get JWT Token: `/api/token/`
  - Refresh Token: `/api/token/refresh/`

## File Explanations:

### 1. `serializers.py`:
- **Purpose**: This file converts Django models (like `User`) into JSON format for the API. It also validates user input and hashes passwords before saving the user.
  
### 2. `views.py`:
- **Purpose**: This file handles the logic for user registration and profile retrieval. It connects the incoming requests (e.g., POST for registration, GET for profile) with the corresponding serializer and returns the appropriate response.

### 3. `urls.py`:
- **Purpose**: This file defines the routes (endpoints) that map to the views in views.py. It ensures that the requests made to /register/ and /profile/ are handled by the correct views.d