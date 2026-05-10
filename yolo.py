from ultralytics import YOLO
import cv2

capture = cv2.VideoCapture(0)

# load YOLO model
model = YOLO("yolo26n-pose.pt")

while True:
    ret,frame = capture.read()

    results = model(frame)
    frame = results[0].plot()

    cv2.imshow("Livestream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()