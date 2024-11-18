from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Evaluate the model
metrics = model.val()
print(metrics)