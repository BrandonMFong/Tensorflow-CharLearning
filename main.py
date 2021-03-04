# Brando 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image 
import os, os.path

model = keras.models.Sequential()
basePathForImages = "B:\\COLLEGE\\20_21\\Spring21\\CES514\\Labs\\Week6\\characters_train\\train\\"
imageFileExtension = ".Bmp"

# print(len([name for name in os.listdir(basePathForImages) if os.path.isfile(name)]))

def task1():
    """
    Task 1:

    Prepare the code to train a convolutional neural network that takes input image of 80x80 and predicts which character they are, i.e. digits 0-9 and upper and lower case letters, a-z, A-Z.
    """
    inputSize = [80,80,1]
    outputSize = (26*2) + 10
    model.add(keras.layers.Input(shape=inputSize))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(outputSize, activation=None))
    model.summary()

def task2():
    """
    Task 2:

    Choose one of the following ways to get the characters dataset:

    - Downloading the First steps with Julia competiton from Kaggle (copy in Week7 folder (Links to an external site.))

    - Downloading the chars74K image dataset (Links to an external site.)

    - Generating your own character dataset using ImageFont (Links to an external site.) in the Pillow package

    and train the network in Task 1 with the images.
    """
    print("Hello")
    imageFiles = np.array(os.listdir(basePathForImages))
    file = basePathForImages + "1" + imageFileExtension
    image = Image.open(file)
    imageArray = np.array(image)
    result = Image.fromarray(imageArray).resize((80,80))

    plt.imshow(result)
    plt.show()

if __name__ == "__main__":
    task1()
    # task2()