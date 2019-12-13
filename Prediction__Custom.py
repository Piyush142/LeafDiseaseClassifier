#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 04:46:15 2019

@author: richguy142
"""
import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from imagenet_utils import preprocess_input
#from keras.models import load_model
#model = load_model('6*3CNN.h')
CATEGORIES = ['Apple__Apple_scab','Apple__Black_rot','Apple__Cedar_apple_rust','Apple__healthy','Blueberry__healthy',
         'Cherry_(including_sour)__healthy','Cherry_(including_sour)__Powdery_mildew','Corn_(maize)__Cercospora_leaf_spotGray_leaf_spot',
         'Corn_(maize)__Northern_Leaf_Blight','Grape__Black_rot','Grape__Esca_(Black_Measles)','Grape__healthy',
         'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)','Orange_Haunglongbing_(Citrus_greening)','Peach__Bacterial_spot'
         ,'Peach__healthy','Pepper,_bell__Bacterial_spot','Pepper,_bell__healthy','Potato__Early_blight','Potato__healthy',
         'Potato__Late_blight','Raspberry__healthy','Soybean__healthy','Squash__Powdery_mildew','Strawberry__healthy',
         'Strawberry__Leaf_scorch','Tomato__Bacterial_spot','Tomato__Early_blight','Tomato__healthy','Tomato__Late_blight',
         'Tomato__Leaf_Mold','Tomato__Septoria_leaf_spot','Tomato__Spider_mitesTwo-spotted_spider_mite','Tomato__Target_Spot',
         'Tomato__Tomato_mosaic_virus','Tomato__Tomato_Yellow_Leaf_Curl_Virus']
# convert class labels to on-hot encoding  # will use this to convert prediction num to string value


def prepare(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

model = tf.keras.models.load_model("Disease.model")
prediction = model.predict([prepare('Patato.jpg')])
# print(prediction)
print(CATEGORIES[np.argmax(prediction)])
