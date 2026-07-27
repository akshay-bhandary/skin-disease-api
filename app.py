from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)

interpreter = tflite.Interpreter(model_path='skin_disease_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Atopic Dermatitis', 'Basal Cell Carcinoma', 'Melanocytic Nevi',
               'Melanoma', 'Warts Molluscum', 'Eczema', 'Benign Keratosis']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img_path = 'temp.jpg'
    file.save(img_path)

    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    os.remove(img_path)
    return jsonify({'prediction': predicted_class, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
