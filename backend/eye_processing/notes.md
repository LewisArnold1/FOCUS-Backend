# Eye Processing App Notes

## Overview:
This app processes live video data to extract eye metrics such as blink count and eye aspect ratio. The backend uses machine learning (maybe) to analyse video frames sent from the frontend and stores these metrics in the database.

### Key Endpoints:
- **Process Eye Data**: `/api/eye/process/` - Accepts video frames and processes them to extract eye metrics.
- **Retrieve Eye Metrics**: `/api/eye/metrics/` - Retrieves the eye metrics (blink count, eye aspect ratio) for the current session.

## File Explanations:

### 1. `serializers.py`:
- **Purpose**: Serializes video data and validates it before processing. It ensures that the incoming video frames are in the correct format.

### 2. `views.py`:
- **Purpose**: Handles requests to process video data and return eye metrics. It connects the incoming data to the processing logic and calculates metrics like blink count and eye aspect ratio.

### 3. `urls.py`:
- **Purpose**: Defines the routes (endpoints) that map to the views in `views.py`. It ensures that requests to `/process/` and `/metrics/` are handled appropriately.

### 4. `processing/`:
- **Purpose**: Contains the logic for processing video frames, including cropping to the eyes and calculating eye metrics like blinks and eye aspect ratio (EAR).

