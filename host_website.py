from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import io
import pickle
import cv2
import numpy as np

app = Flask(__name__,template_folder="templates")



# Load your machine learning model here
# Replace this with your actual model loading code
# Example:
# model = tf.keras.models.load_model('your_model_path')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']

        if image_file:
            # print(image_file)
            image_data = image_file.filename
            # print(image_data)
            # image = Image.open(io.BytesIO(image_data))

            model = pickle.load(open('handwritten_model.pkl', 'rb'))
            img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            img_copy = img.copy()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (400, 440))

            img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
            img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

            img_final = cv2.resize(img_thresh, (28, 28))
            img_final = np.reshape(img_final, (1, 28, 28, 1))

            word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
                         11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
                         21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

            prediction = word_dict[np.argmax(model.predict(img_final))]
            # print(prediction)

            return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
