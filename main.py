# Brando 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image 
import os, os.path
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras import activations
from tensorflow.keras.utils import to_categorical

model = keras.models.Sequential()
basePathForImages = "B:\\COLLEGE\\20_21\\Spring21\\CES514\\Labs\\Week6\\characters_train\\train\\"
trainLabelsFiles = "B:\\COLLEGE\\20_21\\Spring21\\CES514\\Labs\\Week6\\trainLabels.csv"
imageFileExtension = ".Bmp"
inputSize = [80,80,3]
outputSize = (26*2) + 10

# Get Data
labels = pd.read_csv(trainLabelsFiles)
labels['FileExt'] = imageFileExtension
labels["ID"] = labels["ID"].astype(str)
labels["ID"] = labels["ID"].str.cat(labels['FileExt']) 
labels = labels.drop(columns=['FileExt'])

encoder = LabelEncoder()
labels["Target"] = encoder.fit_transform(labels["Class"])

sizeToTrain = 50
numEpochs = 10

# Get the pixels 
limit = sizeToTrain
index = 0
check = False
pixels = np.empty([80,80,3])
for file in labels["ID"]:
    file = basePathForImages + file 
    image = Image.open(file)
    imageArray = np.array(image)
    temp = Image.fromarray(imageArray).resize((80,80))
    train = np.array(temp)

    if check is False:
        pixels = np.concatenate(([pixels], [train]))
        check = True 
    else:
        pixels = np.concatenate((pixels, [train]))

    if index == limit:
        break 
    else:
        index += 1

# print(len([name for name in os.listdir(basePathForImages) if os.path.isfile(name)]))

def task1():
    """
    Task 1:

    Prepare the code to train a convolutional neural network that takes input image of 80x80 and predicts which character they are, i.e. digits 0-9 and upper and lower case letters, a-z, A-Z.
    """
    model.add(keras.layers.Input(shape=inputSize))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(outputSize, activation=activations.softmax))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

def task2():
    """
    Task 2:

    Choose one of the following ways to get the characters dataset:

    - Downloading the First steps with Julia competiton from Kaggle (copy in Week7 folder (Links to an external site.))

    - Downloading the chars74K image dataset (Links to an external site.)

    - Generating your own character dataset using ImageFont (Links to an external site.) in the Pillow package

    and train the network in Task 1 with the images.

    https://www.kaggle.com/c/street-view-getting-started-with-julia/data
        I think I need to get the set and targets from here

        array of images 
    """
    target = to_categorical(labels["Target"].loc[0:sizeToTrain-1], num_classes=outputSize)
    # history = model.fit(pixels[:sizeToTrain], labels["Target"].loc[0:sizeToTrain-1], epochs=numEpochs)
    history = model.fit(pixels[:sizeToTrain], target, epochs=numEpochs)

    test = pixels[40:41]
    prediction = model.predict(test)
    
    # print("Prediction:", np.argmax(prediction))
    print("Prediction:", labels[labels["Target"] == np.argmax(prediction)]["Class"].iloc[0])
    # print("Prediction:", labels[labels["Target"] == 36]["Class"].iloc[0])
    # print(target)
    # print(labels["Target"].loc[0:sizeToTrain-1])
    # print(labels[labels["Target"] == 36]["Class"].iloc[0])

if __name__ == "__main__":
    task1()
    task2()