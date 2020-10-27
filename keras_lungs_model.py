# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:01:06 2020

@author: Vijay
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

import numpy as np
from glob import glob
import matplotlib.pyplot as plt


#Resizing all images to this dimension
IMAGE_SIZE = [224, 224]

train_path = 'archive/chest_xray/train'
test_path = 'archive/chest_xray/test'

# add preprocessing layer 
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights='imagenet', include_top=False)

#dont train existing weights
for layer in vgg.layers:
    layer.trainable=False
    
folders = glob('archive/chest_xray/train/*')

x = Flatten()(vgg.output)


prediction = Dense(len(folders), activation='softmax')(x)

# Model
model = Model(inputs=vgg.input, outputs=prediction)

# Model Structure
model.summary()

model.compile(
        loss = 'categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
)

train_datagen=ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

# Fit the model
r = model.fit_generator(
        training_set,
        validation_data=test_set,
        epochs=5,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set)
 )

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('LossVal_loss') # Save the file, before you show
plt.show()  


# Accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('AccVal_acc') # Save the file before you show
plt.show()



model.save('model_vgg19.h5')
