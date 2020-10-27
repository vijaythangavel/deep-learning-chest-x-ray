# -*- coding: utf-8 -*-

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import numpy as np
model = load_model('model_vgg19.h5')
img = image.load_img('archive/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg', target_size=(224, 224))
#img = image.load_img('archive/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

img_data = preprocess_input(x)
classes = model.predict(img_data)
if classes[0][1] > 0.5:
    print("Pneumonia FOUND")
else:
    print("Pneumonia NOT FOUND")