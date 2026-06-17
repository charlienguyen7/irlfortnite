import mediapipe as mp
import numpy as np
import time
import cv2
import serial
import threading

fps_start = 0
sample_count = 0
trigger = 0
recording = False

# camera dimensions
CAM_WIDTH = 640
CAM_HEIGHT = 480

SEQUENCE_LENGTH = 30

# Set up video camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# set up to read from serial COM7 (Bluetooth)
ser = serial.Serial('COM7', 115200, bytesize=8, parity='N', stopbits=1)
def features_data(result):
    if not result.pose_landmarks:
        return None
    data = []
    for landmarker in result.pose_landmarks[0]:
        data.append(landmarker.x)
        data.append(landmarker.y)
        data.append(landmarker.z)
    return data
def printFPS():
    global fps_start
    fps_end = time.time()
    fps = 1/(fps_end-fps_start)
    fps_start = fps_end
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
def serialRead():
    global ser, trigger
    while(True):
        serialData = ser.read(4)
        trigger = serialData[0]

# Load Mediapipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

serialThread = threading.Thread(target=serialRead, daemon=True)
serialThread.start()

# Configure settings for PoseLandmarker
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

# With configured human pose model as variable 'landmarker'
with PoseLandmarker.create_from_options(options) as landmarker:
    sequence = []
    while cap.isOpened():
        curr_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert OpenCV's BGR color scale to RGB
        rgb_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_cv2)

        # Take timestamp of frame in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)

        # Run prediction model
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Collect output from all 33 keypoints from human pose prediction model
        data = features_data(result)

        if trigger and not recording:
            recording = True

        if recording and data is not None:
            sequence.append(data)
            print(len(sequence))
            if len(sequence) == SEQUENCE_LENGTH:
                print("Done")
                np.save(f"training_data/jumping_{sample_count}.npy", sequence)
                sample_count += 1
                recording = False
                sequence = []

        # If landmark exists then draw circles at each keypoint
        if result.pose_landmarks:
            for landmark in result.pose_landmarks[0]:
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(img=frame, center=(x,y), radius=5, color=(0,255,0), thickness=1)
        
        printFPS()
        cv2.imshow('Mediapipe Feed', frame)    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()