from ultralytics import YOLO
from ultralytics import solutions
from collections import deque
import vgamepad as vg
import cv2
import threading
import time
import math
import serial

# camera dimensions
CAM_WIDTH = 640
CAM_HEIGHT = 360

# global FPS variables
fps_start = 0
fps_end = 0

# global variable declaring current state (STANDING, RUNNING, JUMPING, CROUCHING)
lateralState = "STANDING"
verticalState = "STANDING"

# value from -32768 to 32767
strafeDirection = 0
direction = "STRAIGHT"

RUNNING_THRESHOLD = 0.8 # Time in seconds between left knee and right knee reaching MOVEMENT_THRESHOLD to constitute as "RUNNING"
RUNNING_THRESHOLD_MIN = 0.1 # Minimum time discrepancy between left leg and right leg lifting; meant to prevent crouching from registering as "RUNNING"
RUNNING_ANGLE = 170 # Angle between hips, knees, and ankle to be considered "RUNNING"
JUMPING_THRESHOLD = 0.6 # Normalized y-value of the knees to constitute as "JUMPING"

# global default controller values
trigger = 0 # default is 0 because active-high (positive) logic
joyVX = 0x74
joyVY = 0x7b
joySW = 0

normalizedQuad = 0
shoulderLen = 0

# global variable for camera frame and joint angles
frame = None

leftHip = 0
leftKnee = 0

leftAngle = 0
rightAngle = 0

# initalize XBox 360 gamepad
gamepad = vg.VX360Gamepad()

# set up camera stream
capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# load YOLO model
model = YOLO("yolo26m-pose.pt")

# set up to read from serial COM7 (Bluetooth)
ser = serial.Serial('COM7', 115200, bytesize=8, parity='N', stopbits=1)

# function to print frame rate of camera stream
def printFPS():
    global fps_start
    fps_end = time.time()
    fps = 1/(fps_end - fps_start)
    fps_start = fps_end
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_text, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# function to read data from COM7
def serialRead():
    global trigger, joyVX, joyVY, joySW
    serialData = None
    while True:
        serialData = ser.read(4)
        trigger = serialData[0]
        joyVX = (serialData[1] << 8) - 32768 # gamepad joystick takes in values from -32768 and 32767; serialData[1] is a uint8_t
        joyVY = (serialData[2] << 8) - 32768 # gamepad joystick takes in values from -32768 and 32767; serialData[1] is a uint8_t
        joySW = not serialData[3]
        # print(trigger," ",joyVX," ",joyVY," ",joySW)

# function to display frame onto cv2
def yoloDisplay():
    global frame, leftAngle, rightAngle, lateralState, leftHip, leftKnee, normalizedQuad, shoulderLen, direction
    cv2.putText(frame, "Left: {:.2f}".format(leftAngle), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Right: {:.2f}".format(rightAngle), (0,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, lateralState, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, direction, (0,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    # cv2.putText(frame, "Left Knee: {:.2f}".format(leftKnee[0]), (0,325), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Shoulder Len: {:.2f}".format(shoulderLen), (0,325), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "NZD Quad: {:.2f}".format(normalizedQuad), (0,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    printFPS()
    cv2.imshow("Livestream", frame)

# function that contains YOLO prediction model
def yoloPredict():
    global frame, state, leftHip, leftKnee, leftAngle, rightAngle, normalizedQuad, shoulderLen
    while True:
        ret, frame = capture.read()

        results = model.predict(frame, device=0, verbose=False, max_det=1)
        frame = results[0].plot()

        # extract normalized keypoints from results
        keypoints = results[0].keypoints.xyn

        # you need this because if there is no person in frame, code will crash since 
        # size of first index of keypoints (which represents idx of person) is 0
        try:
            # keypoints for both knees
            leftKnee = keypoints[0][13]
            rightKnee = keypoints[0][14]

            # keypoints for both hips
            leftHip = keypoints[0][11]
            rightHip = keypoints[0][12]
            
            # keypoints for both ankles
            leftAnkle = keypoints[0][15]
            rightAnkle = keypoints[0][16]

            leftShoulder = keypoints[0][5]
            rightShoulder = keypoints[0][6]

            # triangle side lengths of left side
            l1 = math.pow(leftKnee[0]-leftHip[0], 2) + math.pow(leftKnee[1]-leftHip[1], 2)
            l2 = math.pow(leftAnkle[0]-leftKnee[0], 2) + math.pow(leftAnkle[1]-leftKnee[1], 2)
            l3 = math.pow(leftHip[0]-leftAnkle[0], 2) + math.pow(leftHip[1]-leftAnkle[1], 2)

            # triangle side lengths of right side
            r1 = math.pow(rightKnee[0]-rightHip[0], 2) + math.pow(rightKnee[1]-rightHip[1], 2)
            r2 = math.pow(rightAnkle[0]-rightKnee[0], 2) + math.pow(rightAnkle[1]-rightKnee[1], 2)
            r3 = math.pow(rightHip[0]-rightAnkle[0], 2) + math.pow(rightHip[1]-rightAnkle[1], 2)

            # length of torso
            mid_shoulder_y = (leftShoulder[1] + rightShoulder[1]) / 2
            mid_hip_y = (leftHip[1] + rightHip[1]) / 2
            torso_height = abs(mid_shoulder_y - mid_hip_y)

            # length of shoulders
            shoulderLen = math.sqrt((math.pow(leftShoulder[0]-rightShoulder[0], 2) + math.pow(leftShoulder[1]-rightShoulder[1],2)) / torso_height) * 100
            normalizedQuad = (leftKnee[0] - leftHip[0])/l1

            # Law of Cosine to calculate angle between hip, knee, and ankle
            try:
                leftAngle = math.acos((l1 + l2 - l3)/(2*math.sqrt(l1)*math.sqrt(l2))) * 180 / 3.141592653
                rightAngle = math.acos((r1 + r2 - r3)/(2*math.sqrt(r1)*math.sqrt(r2))) * 180 / 3.141592653
            except ValueError:
                pass
    
        except IndexError:
            continue
    
        yoloDisplay()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# function that checks YOLO pose keypoints and determines state of player (i.e. jumping, running)
def movementState():
    global lateralState, verticalState, leftHip, leftKnee, strafeDirection, normalizedQuad, shoulderLen, direction
    leftLegTime = 0
    rightLegTime = 0
    timeOfActivity = 0
    leftStepSeen = False
    rightStepSeen = False
    while True:

        # left step event
        if leftAngle < RUNNING_ANGLE and not leftStepSeen:
            leftLegTime = time.time()
            leftStepSeen = True

        # right step event
        if rightAngle < RUNNING_ANGLE and not rightStepSeen:
            rightLegTime = time.time()
            rightStepSeen = True

        if leftStepSeen and rightStepSeen:
            runningTime = abs(leftLegTime - rightLegTime)
            print("BOTH DETECTED dt:", runningTime)
            if RUNNING_THRESHOLD_MIN < runningTime < RUNNING_THRESHOLD:
                lateralState = "RUNNING"
                timeOfActivity = time.time()
            leftStepSeen = False
            rightStepSeen = False
        
        if shoulderLen < 7:
            if normalizedQuad > 1:
                strafeDirection = -25000
                direction = "LEFT"
            if normalizedQuad < -1:
                strafeDirection = 25000
                direction = "RIGHT"
        else:
            strafeDirection = 0
            direction = "STRAIGHT"
        
        if time.time() - timeOfActivity > RUNNING_THRESHOLD:
            lateralState = "STANDING"

        time.sleep(0.01)


# function that feeds gamepad input

def gamepadInput():
    global gamepad, lateralState, verticalState, strafeDirection, trigger, joyVX, joyVY, joySW
    while True:
        gamepad.right_joystick(joyVX, joyVY)
        
        if verticalState == "JUMPING":
            gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        elif lateralState == "STANDING":
            gamepad.left_joystick(0,0)
        elif lateralState == "RUNNING":
            gamepad.left_joystick(strafeDirection, 32767)

        # if pressed trigger, shoot gun
        if trigger:
            gamepad.right_trigger(255)
        else:
            gamepad.right_trigger(0)

        # if pressed joystick button, reload shotgun
        if joySW:
            gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        else:
            gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            
        gamepad.update()
        
yoloThread = threading.Thread(target=yoloPredict, daemon=True)
serialThread = threading.Thread(target=serialRead, daemon=True)
movementThread = threading.Thread(target=movementState, daemon=True)
gamepadThread = threading.Thread(target=gamepadInput, daemon=True)

yoloThread.start()
serialThread.start()
movementThread.start()
gamepadThread.start()

yoloThread.join()

capture.release()
cv2.destroyAllWindows()

