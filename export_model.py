from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Export the model
model.export(format='onnx')  # Export to ONNX format