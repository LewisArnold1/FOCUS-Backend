# blink_detection Test folder
This folder contains files used to record videos and test the three blink detection methods defined in the Blink Detection paper.
This folder requires the shape_predictor_68_face_landmarks.dat file on the same level to find facial landmarks using dlib.

## Folder Structure
blink_detection Test Folder/
 │-- blink_test_files/ 
 │-- eyes_closed_tests/ 
 │-- SVM_models/
 │-- record-blink-video.py/

**blink_test_files** folder
Folder contains the following:
- All test videos (.avi)
- All corresponding timestamps (.txt)
- EAR values for all videos (.csv)
- Ideal outputs for all videos (.csv)
- Outputs from manual, auto and final svm methods (.csv)
(manual and auto methods contain 3 and 4 columns respectively for the sweep in threshold levels)

**eyes_closed_tests** folder
Folder contains all python files used to calculate EAR values and to test all three methods.

Calculate EAR:
- test_eye_closed.py calculate EAR values for all videos.

Threshold methods:
- process_eye_metrics.py, blinks.py and face.py contain necessary functions for test_eye_closed.py to calculate EARs.
- test_eye_closed.py runs sweeps in both threshold methods and saves outputs to .csv files 
- test_eye_closed.py outputs confusion matrix values & performance metrics.metrics for all thresholds in both sweeps.

SVM method:
- train_SVM is used to train the SVM with varying C and W. It also outputs performance metrics if required.
- test_SVM tests performance metrics with each SVM and saves outputs to .csv files.

# SVM_models
Stores all SVM classifiers created using the scikit learn library in eyes_closed_tests/train_SVM.py

**record_blink_video.py** file
This python file is used by participants to record their 1 minute videos used for testing.
Videos and corresponding timestamps files are saved to the blink_test_files folder.



shape_predictor_68_face_landmarks