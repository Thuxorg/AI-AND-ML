from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Predict on new images
results = model.predict(source='path_to_image_or_video', save=True)