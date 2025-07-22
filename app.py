from flask import Flask, request, jsonify
import easyocr
from ultralytics import YOLO
import os
from PIL import Image

app = Flask(__name__)
model = YOLO("best.pt")  # Replace with your trained model file
reader = easyocr.Reader(['en'])  # Language

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    image_path = os.path.join("uploads", image.filename)
    image.save(image_path)

    results = model(image_path)
    data = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = Image.open(image_path).crop((x1, y1, x2, y2))
            text = reader.readtext(cropped)
            detected_text = ' '.join([t[1] for t in text])
            data.append({'box': [x1, y1, x2, y2], 'text': detected_text})

    os.remove(image_path)
    return jsonify({'predictions': data})

@app.route('/', methods=['GET'])
def home():
    return "Model is working"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
