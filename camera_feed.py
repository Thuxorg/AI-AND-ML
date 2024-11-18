from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

# IP Camera URL
camera_url = "rtsp://username:password@camera_ip_address:port/stream"
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("Error: Could not open camera stream.")
else:
    print("Successfully connected to the camera.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLO model on the frame
    results = model(frame)

    # Render the frame with bounding boxes
    frame_with_boxes = results.render()[0]

    # Display the frame with predictions
    cv2.imshow('YOLO Predictions', frame_with_boxes)

    # Exit loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()