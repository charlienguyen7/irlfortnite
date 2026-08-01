from ultralytics import YOLO
import cv2

# Load YOLO pose model
model = YOLO("yolo26m-pose.pt")   # or yolo11n-pose.pt for faster inference

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run pose detection
    results = model(frame)

    # Draw keypoints and skeleton
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("YOLO Pose", annotated_frame)

    # Quit with q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()