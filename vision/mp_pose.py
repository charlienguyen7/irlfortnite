import mediapipe as mp
import numpy as np
import time
import cv2

# camera dimensions
CAM_WIDTH = 640
CAM_HEIGHT = 480

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models\pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
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

        # If landmark exists then draw circles at each keypoint
        if result.pose_landmarks:
            for landmark in result.pose_landmarks[0]:
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(img=frame, center=(x,y), radius=5, color=(0,255,0), thickness=1)

            cv2.imshow('Mediapipe Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()