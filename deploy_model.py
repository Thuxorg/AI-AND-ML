import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession('best.onnx')

# Perform inference
input_image = preprocess_image('path_to_image.jpg')  # Preprocess the input
outputs = session.run(None, {'input': input_image})
print(outputs)