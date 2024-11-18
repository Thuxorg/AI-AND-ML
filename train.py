from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Choose a YOLOv8 model variant

# Train the model
model.train(data='dataset/disease_data.yaml', epochs=50, batch=16, imgsz=640)