# Face Pose Test Folder

This folder contains various tests conducted for evaluating face pose estimation methods in isolation from the main backend. The tests include implementations using the **dlib** and **MediaPipe** libraries, along with integration tests to ensure seamless incorporation into the main system.

## Folder Structure
Face Pose Test Folder/ 
│-- dlib_landmark_tests/ 
│-- mp_landmark_tests/ 
│-- integration_test/ 
│-- integration_test_2

### dlib_landmark_tests
This folder contains tests using the original **dlib** library for face landmark detection. The tests evaluate **dlib’s** effectiveness in tracking facial landmarks and analysing face pose.

### mp_landmark_tests
This folder contains tests integrating **MediaPipe** for face landmark detection. These tests assess **MediaPipe’s** performance and accuracy in face pose estimation while working in isolation from the main backend.

### Integration Tests
- **integration_test:** Ensures that the selected face pose estimation method integrates correctly into the main system.
- **integration_test_2:** A secondary integration test to verify compatibility and resolve potential issues in merging the implementation into the backend.

## Purpose
These tests were conducted to compare **dlib** and **MediaPipe** for face pose estimation and ensure smooth integration into the main backend. The integration tests validate the successful incorporation of the selected **MediaPipe** approach.

