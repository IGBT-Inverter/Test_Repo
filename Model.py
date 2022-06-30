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

"""

DATADIR = "C:/Users/Owner/Desktop/Lego_Data_Version_1/LegoClasses"
CATEGORIES = ["2639", "3009", "3021", "3034", "3308", "3666", "3710", "41769"]

IMG_SIZE = 256 #This is the normalized image size in pixels (FOR SQUARE-IMAGES CHANGE TO DESIRED SIZE)

training_data = [] #creates empty training data list to be appeneded into

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #navigates path to proper folder directory
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #peforms greyscale conversion
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #resizes each image in the for loop
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data)) # = 1728 images 

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

x_train = []
y_train = []

for features, label in training_data: 
    x_train.append(features) #feature would be the image
    y_train.append(label) #label would be the 0 or 1

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train) 

pickle_out = open("x_lego.pickle", "wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("y_lego.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

"""

x_train = pickle.load(open("x_lego.pickle", "rb"))
y_train = pickle.load(open("y_lego.pickle", "rb"))

#Code below automatically splits the input Lego images into two sets and assigns it to the 
#variables x_train, x_test, y_train and y_test with a 20% split size

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#plt.imshow(x_train[7], cmap="gray")
#plt.show()

x_train = x_train/255.0
x_test = x_test/255.0

#Function below returns the uncompiled (aka untrained neural network) model object
#Needs to be provided with "x_train" when x_train is a numpy array so as to allow
#the first layer to properly match the input shape of the input data

def conv_net(x_train):
    
    model = Sequential()
    model.add(Conv2D(128, (3,3), input_shape = x_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128))

    model.add(Dense(8))
    model.add(Activation('softmax'))

    return model

#Function below compiles the neural network
#Only requires the uncompiled model to be passed

def compiling_neural(model):
    
    loss = keras.losses.SparseCategoricalCrossentropy()
    optim = keras.optimizers.Adam()
    metrics = [keras.metrics.SparseCategoricalAccuracy()]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    
    return model

#Function just saves model in tensorflow standard format under name "neurally_net"

def saving_model(model):
    model.save("neurally_net")

#Function below trains the model on the training set, then evaluates performance on testing set
#Requires compiled model
#Returns history object so that plot of accuracy and loss can be generated in Jupyter Notebook

def evaluate_neural(model):
    
    history = model.fit(x_train, y_train, batch_size=32, epochs=4)
    model.evaluate(x_test, y_test, batch_size=64, verbose=2)
    saving_model(model) #Will save the model weights to WORKING DIRECTORY
    
    return history 

model = conv_net(x_train)
model = compiling_neural(model)
history = evaluate_neural(model)