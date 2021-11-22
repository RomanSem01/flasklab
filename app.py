from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.applications import resnet50
import os

UPLOAD_FOLDER = 'images/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def recognize_squirrel(filename):
    model = load_model('model.h5')
    img = cv2.imread(f'images/{filename}')
    img = cv2.resize(img, (int(100), int(100)))
    img = resnet50.preprocess_input(img)
    img = img.reshape(1, 100, 100, 3).astype('float32')
    x = model.predict(img)
    if np.argmax(x, axis=1)[0] == 5:
        return 'It is a squirrel'
    else:
        return 'It is not a squirrel'



@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    img = request.files['img-file']
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], img.filename))
    return render_template('result.html', result=recognize_squirrel(img.filename))



if __name__ == '__main__':
    app.run()
