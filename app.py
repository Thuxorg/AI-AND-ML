from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

# Camera URL
camera_url = "rtsp://username:password@camera_ip_address:port/stream"
cap = cv2.VideoCapture(camera_url)

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = model(frame)

        # Render frame with bounding boxes
        frame_with_boxes = results.render()[0]

        # Convert frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)