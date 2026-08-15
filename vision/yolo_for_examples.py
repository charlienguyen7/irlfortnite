from ultralytics import YOLO
from ultralytics import solutions
from collections import deque
import vgamepad as vg
import cv2
import threading
import time
import math
import serial

# rep data
REP_ANGLE = 70
MIN_REP_ANGLE = 150
repCounter = 0
repDone = False
rightAngle = 180

# camera dimensions
CAM_WIDTH = 640
CAM_HEIGHT = 360

# global FPS variables
fps_start = 0
fps_end = 0

# global variable for camera frame and joint angles
frame = None

# set up camera stream
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# load YOLO model
model = YOLO("yolo26m-pose.pt")

# function to print frame rate of camera stream
def printFPS():
    global fps_start
    fps_end = time.time()
    fps = 1/(fps_end - fps_start)
    fps_start = fps_end
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_text, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# function to display frame onto cv2
def yoloDisplay():
    global frame, rightAngle, repCounter
    cv2.putText(frame, "Angle: {:.2f}".format(rightAngle), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Reps: {:.2f}".format(repCounter), (0,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    printFPS()
    cv2.imshow("Livestream", frame)

# function that contains YOLO prediction model
def yoloPredict():
    global frame, rightAngle
    while True:
        ret, frame = capture.read()

        results = model.predict(frame, device=0, verbose=False, max_det=1)
        frame = results[0].plot()

        # extract normalized keypoints from results
        keypoints = results[0].keypoints.xyn

        # you need this because if there is no person in frame, code will crash since 
        # size of first index of keypoints (which represents idx of person) is 0
        try:
            # keypoints for arm joints (right arm)
            rightShoulder = keypoints[0][6]
            rightElbow = keypoints[0][8]
            rightWrist = keypoints[0][10]

            # triangle side lengths of left side
            r1 = math.pow(rightShoulder[0]-rightElbow[0], 2) + math.pow(rightShoulder[1]-rightElbow[1], 2)
            r2 = math.pow(rightElbow[0]-rightWrist[0], 2) + math.pow(rightElbow[1]-rightWrist[1], 2)
            r3 = math.pow(rightShoulder[0]-rightWrist[0], 2) + math.pow(rightShoulder[1]-rightWrist[1], 2)

            # Law of Cosine to calculate angle between hip, knee, and ankle
            try:
                rightAngle = math.acos((r1 + r2 - r3)/(2*math.sqrt(r1)*math.sqrt(r2))) * 180 / 3.141592653
            except ValueError:
                pass
    
        except IndexError:
            continue
    
        yoloDisplay()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# function that checks YOLO pose keypoints and determines state of player (i.e. jumping, running)
def repState():
    global rightAngle, repCounter, repDone
    while True:
        # rep event
        if rightAngle < REP_ANGLE and not repDone:
            repCounter += 1
            repDone = True
        if repDone and rightAngle > MIN_REP_ANGLE:
            repDone = False
        time.sleep(0.01)

        
yoloThread = threading.Thread(target=yoloPredict, daemon=True)
repThread = threading.Thread(target=repState, daemon=True)

yoloThread.start()
repThread.start()

yoloThread.join()

capture.release()
cv2.destroyAllWindows()

