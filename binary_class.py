

#create a deep learning model to work on

def model_M1():
    input_shape = (64, 64, 3)
     #single convolutional layer followed by a pooling layer
    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), padding = "same", input_shape = input_shape, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(16, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(32, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model
    
    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

from glob import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def preprocess_dataset(dataDir, labelDictionary):
  x = []
  y = []
  
  
  width, height = 64, 64
  
  # list folders in directory 
  directory = os.listdir(dataDir)
  
  #for each folder (train and validation)
  for label in directory:
    
    #add class label to label dictionary
    if label not in labelDictionary:
      labelDictionary[label] = len(labelDictionary)
      
    # create full path for image directory 
    #(append absolute and image directory path)
    sourceImages = os.path.join(dataDir, label)
    images = os.listdir(sourceImages)
    
    #for each image in directory
    for image in images:
      
      #read the image from file, resize and add to a list
      full_size_image = cv2.imread(os.path.join(sourceImages, image))
      x.append(cv2.resize(full_size_image, (width, height), 
                         interpolation = cv2.INTER_CUBIC))
      
      #add the class label to y
      y.append(label)
      
  return np.array(x), np.array(y)


def main():

  width, height = 64, 64

  # mount the dataset from the google drive or from your hard-drive
  # specify the path for train data and validation data
  trainDataDir =  '/content/gdrive/My Drive/keras_lab_dataset/dogs_cats/dogs_cats/data/train'
  validationDataDir = '/content/gdrive/My Drive/keras_lab_dataset/dogs_cats/dogs_cats/data/validation'

  NUM_EPOCHS = 20
  batch_size = 64

  # this dictionary will contain all lables with an associated int value
  labelDictionary = {1:'cat', 2:'dog'}

  # the preprocess function will read all image data from a folder
  # convert it to a Numpy array and add to a list
  # it will also add the label for the image, the folder name

  trainImages, trainLabels = preprocess_dataset(trainDataDir, labelDictionary)

  valImages, valLabels = preprocess_dataset(validationDataDir, labelDictionary)
  
  # Normalize the data
  trainImages = trainImages.astype("float")/255.0
  valImages = valImages.astype("float")/255.0

  #Map string label values to integer values
  trainLabels = (pd.Series(trainLabels).map(labelDictionary)).values
  valLabels = (pd.Series(valLabels).map(labelDictionary)).values
    
  print("Compiling model...")
  opt = tf.keras.optimizers.SGD(lr=0.01)
  model1 = model_M1()
  
  model1.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
 
  trainDataGenerator = ImageDataGenerator(shear_range = 0.2,
                                          zoom_range = 0.2,
                                          rotation_range = 30,
                                          horizontal_flip = True)

  train_generator = trainDataGenerator.flow(trainImages,
                                          trainLabels, batch_size = 64)
   
#   print (model1.summary())
    
  print("Training network ModelM1..... ", )
  H1 = model1.fit_generator(train_generator, 
                        validation_data = (valImages, valLabels),
                        steps_per_epoch = len(trainImages)/batch_size,
                        epochs = NUM_EPOCHS)
    
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["loss"], label="train_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["acc"], label="train_acc")
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.show()
  
  
  
  
# call the main function
main()
