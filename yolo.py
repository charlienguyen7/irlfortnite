from ultralytics import YOLO
import vgamepad as vg
import cv2
import time

# initalize XBox 360 gamepad
gamepad = vg.VX360Gamepad()

buttonTime = 0.2
buttonPressed = 0

# camera dimensions
CAM_WIDTH = 640
CAM_HEIGHT = 360

# global FPS variables
fps_start = 0
fps_end = 0

# global variable declaring current state (STANDING, RUNNING, JUMPING, CROUCHING)
state = "STANDING"
STATE_DELAY = 0.5 # Delay between each state meant to combat constant switching
MOVEMENT_THRESHOLD = 0.35 # Difference between hips and knees to constitute as "MOVEMENT"
RUNNING_THRESHOLD = 0.7 # Time in seconds between left knee and right knee reaching MOVEMENT_THRESHOLD to constitute as "RUNNING"
JUMPING_THRESHOLD = 0.45 # Normalized y-value of the knees to constitute as "JUMPING"

leftKneeTime = 0
rightKneeTime = 0
lastActivityTime = time.time()

# function to print frame rate of camera stream
def printFPS():
    global fps_start
    fps_end = time.time()
    fps = 1/(fps_end - fps_start)
    fps_start = fps_end
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_text, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# load YOLO model
model = YOLO("yolo26n-pose.pt")

while True:
    ret,frame = capture.read()

    results = model(frame)
    frame = results[0].plot()

    # extract normalized keypoints from results
    keypoints = results[0].keypoints.xyn

    # you need this because if there is no person in frame, code will crash since 
    # size of first index of keypoints (which represents idx of person) is 0
    try:
        leftKnee = keypoints[0][13][1]
        rightKnee = keypoints[0][14][1]

        leftHip = keypoints[0][11][1]
        rightHip = keypoints[0][12][1]
    except IndexError:
        continue
    
    leftDisplacement = abs(leftKnee - leftHip)
    rightDisplacement = abs(rightKnee - rightHip)

    if leftDisplacement < MOVEMENT_THRESHOLD:
        leftKneeTime = time.time()
    if rightDisplacement < MOVEMENT_THRESHOLD:
        rightKneeTime = time.time()

    if time.time() - leftKneeTime > RUNNING_THRESHOLD:
        leftKneeTime = 0
    if time.time() - rightKneeTime > RUNNING_THRESHOLD:
        rightKneeTime = 0
    
    if (leftKneeTime != 0 and rightKneeTime != 0) and (abs(leftKneeTime - rightKneeTime) < RUNNING_THRESHOLD):
        state = "RUNNING"
        lastActivityTime = time.time()
    
    if leftKnee < JUMPING_THRESHOLD and rightKnee < JUMPING_THRESHOLD:
        state = "JUMPING"
        lastActivityTime = time.time()

    if time.time() - lastActivityTime >= STATE_DELAY:
        state = "STANDING"
    
    cv2.putText(frame, state, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    leftKneeText = "Left Knee: {:.2f}".format(leftKnee)
    rightKneeText = "Right Knee: {:.2f}".format(rightKnee)
    leftHipText = "Left Hip: {:.2f}".format(leftHip)
    rightHipText = "Right Hip: {:.2f}".format(rightHip)
    # cv2.putText(frame, leftHipText, (0,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    # cv2.putText(frame, rightHipText, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    printFPS()
    cv2.imshow("Livestream", frame)

    if state == "JUMPING":
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        gamepad.update()
        buttonPressed = time.time()
    elif state == "STANDING":
        gamepad.left_joystick(0,0)
        gamepad.update()
    elif state == "RUNNING":
        gamepad.left_joystick(0, 32767)
        gamepad.update()
    
    if time.time() - buttonPressed > buttonTime:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        gamepad.update()
        buttonPressed = 0
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

