import numpy as np
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, Dropout
from keras.preprocessing.image import ImageDataGenerator
import sklearn
import matplotlib.image as mpimg
import cv2
import glob
from random import randint

batch_size = 80
epochs = 20
csv_file = open("./data/labels.csv")
images_cars_dir = "./data/cars/*.jpg"
images_not_cars_dir = "./data/not_cars/*.jpg"


images_cars = glob.glob(images_cars_dir)
images_not_cars = glob.glob(images_not_cars_dir)
print("number of car images: " + str(len(images_cars)))
print("number of non car images: " + str(len(images_not_cars)))
images = images_cars + images_not_cars
print("number of images: " + str(len(images)))

n_images_cars = 8128  # read from file explorer
n_images_not_cars = 8968  # read from file explorer
labels = n_images_cars * [1] + n_images_not_cars * [0]  # car has label 1 and not car label 0
n_samples = n_images_cars + n_images_not_cars



def random_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image[:, :, 2] = image[:, :, 2] * random_bright
    image[:, :, 2][image[:, :, 2] > 255] = 255
    image = np.array(image, dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def random_shift(image):
    #shift the traffic sign randomly in the image
    rows, cols, depth = image.shape
    M = np.float32([[1,0,randint(-10, 10)],[0,1,randint(-10, 10)]])
    return cv2.warpAffine(image,M,(cols,rows))

def generator(images, labels, batch_size=32):
    num_samples = len(images)
    images, labels = sklearn.utils.shuffle(images, labels)
    while 1:  # Loop forever so the generator never terminates
        images, labels = sklearn.utils.shuffle(images, labels)
        for offset in range(0, num_samples, batch_size):
            batch_samples = images[offset:offset+batch_size]
            images_batch = []
            labels_batch = []
            for i in range(len(batch_samples) - 1):
                # load images
                #image = mpimg.imread(images[i])
                image = mpimg.imread(images[i])
                image = random_shift(image)
                image = random_brightness(image)
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image_flipped = cv2.flip(image, 1)
                images_batch.append(image)
                labels_batch.append(labels[i])
                images_batch.append(image_flipped)
                labels_batch.append(labels[i])

            X_train = np.array(images_batch)
            #X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
            y_train = np.array(labels_batch)
            yield sklearn.utils.shuffle(X_train, y_train)


images, labels = sklearn.utils.shuffle(images, labels)
training_split_index = int(n_samples*0.9)
n_training_samples = training_split_index + 1
n_validation_samples = n_samples - n_training_samples

print("number of training images: " + str(n_training_samples))
print("number of validation images: " + str(n_validation_samples))
print("Index to split the images: " + str(training_split_index))

x_train = []
for i in range(training_split_index):
    x_train.append(cv2.imread(images[i]))

x_train = np.array(x_train)
y_train = np.array(labels[0:training_split_index])

training_generator = generator(images[0:training_split_index], labels[0:training_split_index], batch_size=batch_size)
validation_generator = generator(images[training_split_index: n_samples-1], labels[training_split_index: n_samples-1], batch_size=batch_size)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)


model = Sequential([
    Lambda(lambda x: (x / 128), input_shape=(64, 64, 3)),
    Conv2D(filters=9, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation="relu", input_shape=(64, 64, 3)),
    Dropout(0.25),
    Conv2D(filters=18, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation="relu"),
    Dropout(0.25),
    Conv2D(filters=30, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation="relu"),
    Dropout(0.25),
    Conv2D(filters=42, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation="relu"),
    Flatten(),
    Dropout(0.5),
    Dense(30, activation="relu"),
    #Dropout(0.5),
    #Dense(8, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(training_generator, validation_data=validation_generator, validation_steps=n_validation_samples/batch_size, epochs=epochs, steps_per_epoch=n_training_samples/batch_size)
model.save("model.h5")
