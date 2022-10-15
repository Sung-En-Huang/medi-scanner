import numpy as np
import cv2
import tensorflow
from tensorflow import keras

# from sklearn.datasets import make_regression, make_classification, make_blobs
# import pandas as pd
# import matplotlib.pyplot as plt

# # TensorFlow and tf.keras
# import tensorflow.compat.v1 as tf

# from tensorflow import keras

# # Helper libraries
# import numpy as np
# import matplotlib.pyplot as plt

# print(tf.__version__)

# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

# Importing all necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_width, img_height = 224, 224

train_data_dir = 'C:/Users/Owner/Documents/Jeffrey/Hack The Valley/medi-scanner/Data/Train'
validation_data_dir = 'C:/Users/Owner/Documents/Jeffrey/Hack The Valley/medi-scanner/Data/Validation'
nb_train_samples =36
nb_validation_samples = 7
epochs = 10
batch_size = 20

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
  
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
  
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
  
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
  
test_datagen = ImageDataGenerator(rescale=1. / 255)
  
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
  
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
  
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


model.save_weights('model_saved.h5')

from keras.models import load_model
from keras.preprocessing.image import load_img ### HERE

from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
  
from keras.models import load_model
  
model = load_model('model_saved.h5')
  
image = load_img('C:/Users/Owner/Documents/Jeffrey/Hack The Valley/medi-scanner/Data/Testing Image.jpg', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)
print("Predicted Class (0 - First Degree Burn , 1- Healthy Skin): ", label[0][0])

# DATADIR = "C:/Users/Owner/Documents/Jeffrey/Hack The Valley/medi-scanner/Data"

# CATEGORIES = ["First Degree", "Second Degree", "Third Degree", "Healthy Skin"]

# for category in CATEGORIES: # do 1st, 2nd, 3rd
#     path = os.path.join(DATADIR,category) 
#     for img in os.listdir(path):  # iterate over each image per dogs and cats
#         img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
#         plt.imshow(img_array, cmap='gray')  # graph it
#         plt.show()  # display!

#         IMG_SIZE = 200

#         new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#         plt.imshow(new_array, cmap='gray')
#         plt.show()

#         break
#     break


# def main():

#     # ==============================================================
#     # DISPLAY VIDEO WINDOW
#     # ==============================================================
#     cap = cv2.VideoCapture(0)
#     ptime = 0
#     ctime = 0

#     while True:
#         success , img = cap.read()
#         img_iso = np.empty(img.shape)
#         img_iso.fill(0)

#         cv2.imshow("Capture" , img)

#         if cv2.waitKey(1) &0xFF == ord('x'):
#             break # Press 'x' to close window

# if __name__ == '__main__':
#     main()