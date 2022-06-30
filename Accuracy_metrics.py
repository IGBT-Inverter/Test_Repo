import os
from matplotlib.cbook import flatten
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from turtle import shape
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import random
from multiprocessing import pool
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle 
from sklearn.model_selection import train_test_split

new_model = keras.models.load_model("neurally_net")
new_model.summary()

x_train = pickle.load(open("x_lego.pickle", "rb"))
y_train = pickle.load(open("y_lego.pickle", "rb"))

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20)

x_train = x_train/255.0
x_test = x_test/255.0

new_model.evaluate(x_test, y_test, verbose=2)

y_pred = np.argmax(new_model.predict(x_test), axis=-1)

print(y_pred)

print(y_test)

# Function for accuracy metric 
# Displays the softmax activation probability for classification 

def bounding_accuracy(image, model):
    img = (np.expand_dims(image,0))
    my_prediction = model.predict(img)
    
    num_pred = my_prediction[0][0]
    
    length = len(my_prediction[0])

    for i in range(len(my_prediction[0])):
        if my_prediction[0][i] > num_pred:
            num_pred = my_prediction[0][i]
    
    return num_pred

def predicted_class(image, model):
    img = (np.expand_dims(image,0))
    my_prediction = model.predict(img)
    
    int_range = np.argmax(my_prediction)
    
    prediction = 0
    
    if int_range == 0:
        prediction = 2639
    elif int_range == 1:
        prediction = 3009
    elif int_range == 2: 
        prediction = 3021
    elif int_range == 3:
        prediction = 3034
    elif int_range == 4: 
        prediction = 3308
    elif int_range == 5:
        prediction = 3666
    elif int_range == 6:
        prediction = 3710
    elif int_range == 7:
        prediction = 41769

    return prediction
        
plt.imshow(x_train[10], cmap="gray")
plt.show()

bounding = bounding_accuracy(x_train[10], new_model)
predicted_lego = predicted_class(x_train[10], new_model)

print("")
print(bounding)
print(predicted_lego)