#Manual Libraries Installation -- Tensorflow, Keras, OpenCV, Flask, NumPy, Socket.io, Eventlet

# Import Required library
import socketio            # connect model to simulator
import eventlet            # listen request
from flask import Flask    # Web framwork for PYTHON
import numpy as np         # Numerical Python library
import tensorflow.keras as keras
from keras.models import load_model  # Keras load odel
import base64              # decode base64 iage
from io import BytesIO     # buffer
from PIL import Image       #python image library
import cv2

sio = socketio.Server()    # Bi Directional connection b/w client and server
app = Flask(__name__)      # main

# DEMO

"""
@app.route('/home')
def greeting():
    return 'Welcome!'
"""
# Setting speed limit variable
speed_limit = 10

# Input Image Frame Preprocessing
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# EVENT HANDLING
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)


# Connection to Simulation
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

# Send vehicle Control data to model
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

# Run main code
if __name__ == '__main__':
    #app.run(port=4000)
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)  #'' means ip
