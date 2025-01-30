import cv2
import time
import subprocess
import csv
import threading
import os

# Define constants
VIDEO_FILENAME = "name_test_x.avi"  # change to your name + test number x
CSV_FILENAME = "name_test_x.csv"
VIDEO_DURATION = 10  # in seconds - change to 60!? or more
FRAME_RATE = 30
FRAME_SIZE = (640, 480)


def record_video(video_filename, duration, frame_rate, frame_size):
    """
    Records a video for the specified duration and saves it to a file.
    """
    cap = cv2.VideoCapture(0)  # Capture from the default camera

    # don't set these? - just record what they are maybe
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    cap.set(cv2.CAP_PROP_FPS, frame_rate) # we dont want to set the frame rate surely?
    # we are choosing to record timestampt of each frame so frame rate is irrelevant?

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, frame_rate, frame_size)

    print("Recording video...")
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print("Video recording complete.")

'''
def run_eye_test(csv_filename):
    """
    Runs the external eye processing script and saves the output (1s and 0s) into a CSV file.
    """
    command = ["python", "./eye_processing/test_eye_metrics.py", VIDEO_FILENAME]

    print("Running eye processing test...")
    with open(csv_filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Eye State"])

        # Execute the eye processing script and capture real-time output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            line = line.strip()
            if line.isdigit():  # Ensure the output is either 0 or 1
                writer.writerow([line])

        process.stdout.close()
        process.wait()
    print("Eye processing test complete.")
'''

def play_video(video_filename):
    """
    Plays the recorded video.
    """
    cap = cv2.VideoCapture(video_filename)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    print("Playing back video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Playback', frame)
        # Break the loop on pressing 'q'
        if cv2.waitKey(int(1000 / FRAME_RATE)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # Record the video in a separate thread to allow real-time processing if needed
    video_thread = threading.Thread(target=record_video, args=(VIDEO_FILENAME, VIDEO_DURATION, FRAME_RATE, FRAME_SIZE))
    video_thread.start()
    video_thread.join()

    # Run the eye processing script and store output concurrently
    eye_test_thread = threading.Thread(target=run_eye_test, args=(CSV_FILENAME,))
    eye_test_thread.start()

    # Play the video while the eye processing test is running
    play_video(VIDEO_FILENAME)
    
    eye_test_thread.join()


if __name__ == "__main__":
    main()
