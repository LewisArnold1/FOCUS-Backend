import cv2
import time
import subprocess
import csv
import threading
import os

# change to your name + test number x
VIDEO_FILENAME = "name_test_x.avi"
TIMESTAMP_FILENAME = "name_test_x_timestamps.t"

# Set to 60s for recording videos (can use 10s if you want to test its working)
VIDEO_DURATION = 10  

# CSV_FILENAME = "name_test_x.csv"  # csv file output in test_eye_metrics

def record_video(video_filename, timestamp_filename, duration):#, frame_rate, frame_size):
    """
    Records a video for the specified duration and saves it to a file.
    """
    cap = cv2.VideoCapture(0)  # Capture from the default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get frame width and height for saving video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    print("Recording video...")
    timestamps = []
    frames = []
    start_time = time.time()
   
    # Record Video & save timestamps 
    with open(timestamp_filename, "w") as f:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # Save frame
            frames.append(frame)

            # Save timestamp
            timestamp = time.time() - start_time
            timestamps.append(timestamp)
            f.write(f"{timestamp:.6f}\n")
    cap.release()

    if len(timestamps) < 2:
        print("Error: Not enough frames recorded.")
        return

    # Calculate fps of recorded video
    avg_fps = len(timestamps) / (timestamps[-1] - timestamps[0])

    # Save video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, avg_fps, (frame_width,frame_height))
    for frame in frames:
        out.write(frame)
    out.release()

    print(f"Video saved: {video_filename} ({timestamps[-1]:.2f} seconds, {avg_fps:.2f} FPS).")


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


# include below
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
'''


'''
# def main():
    
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
    

# if __name__ == "__main__":
#     main()

'''

record_video(VIDEO_FILENAME,TIMESTAMP_FILENAME,VIDEO_DURATION)
