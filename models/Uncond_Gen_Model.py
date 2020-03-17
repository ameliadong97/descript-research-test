#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
import sys 

sys.path.insert(0,'..')
from utils import plot_stroke

import numpy as np
strokes = np.load('../data/strokes-py3.npy', allow_pickle=True)

# Due to time constraint, will not train for the whole dataset
import math
training = strokes[:math.floor(len(strokes)*0.05)]

# Normalize training co-ordinate offsets
x_mean, y_mean, count = 0, 0, 0

for stroke in training:
    for i in stroke:
        x_mean += i[1]
        y_mean += i[2]
        count += 1
x_mean /= count
y_mean /= count

std_x, std_y = 0, 0
for stroke in training:
    for i in stroke:
        std_x += (i[1]-x_mean)**2
        std_y += (i[2]-y_mean)**2
std_x /= count
std_y /= count
std_x = std_x**(0.5)
std_y = std_y**(0.5)

for stroke in training:
    for i in stroke:
        i[1] = (i[1]-x_mean)/std_x
        i[2] = (i[2]-y_mean)/std_y

# Prepare training data as X and y.
# Each sample of X is of shape (400,3) and each sample of y is of shape (1,3)
# i.e. use the first 400 strokes to predict the last one
X = []
y = []
for sample in training:
    for i in range(len(sample)-400-2):
        X.append(sample[i:i+400])
        y.append(sample[i+400+1])
X = np.array(X)
y = np.array(y)

from tensorflow import keras
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras.models import Model
import mdn


def build_up_model():    
    keras.backend.clear_session()
    
    inputs = Input(shape=(400,3))
    x = LSTM(256, return_sequences=True,batch_input_shape = (None,400,3))(inputs)
    x = LSTM(256)(x)
    outputs = mdn.MDN(3, 10)(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss=mdn.get_mixture_loss_func(3,10), optimizer=keras.optimizers.Adam())
    
    # Fit the model
    history = model.fit(X, y, batch_size=128, epochs=10, validation_split = 0.2)
    
    model.save_weights('model_weights.h5')

def generate_unconditionally(random_seed=1):
    np.random.seed(random_seed)
    
    # Set up the generator
    inputs = Input(shape=(1,3))
    x = LSTM(256, return_sequences=True,batch_input_shape = (1,1,3))(inputs)
    x = LSTM(256)(x)
    outputs = mdn.MDN(3, 10)(x)
    generator = Model(inputs=inputs,outputs=outputs)
    generator.compile(loss=mdn.get_mixture_loss_func(3,10), optimizer=keras.optimizers.Adam())
    generator.load_weights('model_weights.h5')
    
    predictions = []
    stroke_pt = np.asarray([1,0,0], dtype=np.float32) # start point
    predictions.append(stroke_pt)

    for i in range(400):
        stroke_pt = mdn.sample_from_output(generator.predict(stroke_pt.reshape(1,1,3))[0], 3, 10)
        predictions.append(stroke_pt.reshape((3,)))
        
    predictions = np.array(predictions, dtype=np.float32)
    for i in range(len(predictions)):
        predictions[i][0] = (predictions[i][0] > 0.5)*1
        predictions[i][1] = predictions[i][1] * std_x + x_mean
        predictions[i][2] = predictions[i][2] * std_y + y_mean
    return predictions

#stroke = generate_unconditionally()
#plot_stroke(stroke)