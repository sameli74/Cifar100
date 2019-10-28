# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import pickle
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input, merge, Add
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import np_utils
import math
from keras.callbacks import LearningRateScheduler
from keras.models import Model,load_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
batch_size = 128
num_classes = 100
epochs = 120
addr='../input/'
np.random.seed(2017)
%matplotlib inline

# Any results you write to the current directory are saved as output.

def read_data():
    with open(addr +'train_data/train_data', 'rb') as f:
        x_train = pickle.load(f)
        y_train= pickle.load(f)
    with open(addr + 'test_data/test_data', 'rb') as f:
        x_test = pickle.load(f)
    return x_train, y_train, x_test
def prepare_data(x_train, y_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train=np.reshape(x_train, (-1, 3, 32, 32)).transpose(0,2,3,1)
    y_train = np_utils.to_categorical(y_train, num_classes)
    x_test=np.reshape(x_test, (-1, 3, 32, 32)).transpose(0,2,3,1)
    return x_train, y_train, x_test

def visualize_data(x_train):
    plt.figure()
    fig_size = [20, 20]
    plt.rcParams["figure.figsize"] = fig_size
    for i in range(0,100):
        ax = plt.subplot(10, 10, i+1)
        img = x_train[i,:,:,:]
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(img)
    plt.show()

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def printGraph(history):

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def save_csv(results):
    results = pd.Series(results,name="labels")

    # Add image id column
    submission = pd.concat([pd.Series(range(10000),name = "ids"),results],axis = 1)
    print(submission)
    # Export to .csv
    submission.to_csv("my_predictions.csv",index=False)

def create_model():
    inp=Input((32, 32, 3))
    x=Conv2D(16, (3, 3), padding='same',data_format='channels_last',
                     input_shape=x_train.shape[1:])(inp)
    x= BatchNormalization()(x)
    x= Activation('elu')(x)

    y= Conv2D(128, (1, 1), padding='same')(x)
    x= Conv2D(128, (3, 3), padding='same')(x)
    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= Conv2D(128, (3, 3), padding='same')(x)
    x= Add()([x,y])

    y= x
    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= Conv2D(128, (3, 3), padding='same')(x)
    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= Dropout(0.3)(x)
    x= Conv2D(128, (3, 3), padding='same')(x)
    x= Add()([x,y])


    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= MaxPooling2D(pool_size=(2, 2))(x)


    y= Conv2D(256, (1, 1), padding='same')(x)
    x= Conv2D(256, (3, 3), padding='same')(x)
    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= Conv2D(256, (3, 3), padding='same')(x)
    x= Add()([x,y])


    y= x
    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= Conv2D(256, (3, 3), padding='same')(x)
    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= Dropout(0.3)(x)
    x= Conv2D(256, (3, 3), padding='same')(x)
    x= Add()([x,y])
    # x= MaxPooling2D(pool_size=(2, 2))(x)

    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= MaxPooling2D(pool_size=(2, 2))(x)

    y= Conv2D(512, (1, 1), padding='same')(x)
    x= Conv2D(512, (3, 3), padding='same')(x)
    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= Conv2D(512, (3, 3), padding='same')(x)
    x= Add()([x,y])

    y= x
    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= Conv2D(512, (3, 3), padding='same')(x)
    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= Dropout(0.3)(x)
    x= Conv2D(512, (3, 3), padding='same')(x)
    x= Add()([x,y])

    x= BatchNormalization()(x)
    x= Activation('elu')(x)
    x= AveragePooling2D(pool_size=(8, 8))(x)

    x= Flatten()(x)
#     x= Dense(512)(x)
    x= Activation('elu')(x)
    x= Dropout(0.3)(x)
    x= Dense(num_classes)(x)
    outp= Activation('softmax')(x)


    model= Model(inp, outp)
    return model

def run():
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]
    model.summary()
    datagen.fit(x_train)
    history=model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size=batch_size),
                        callbacks=callbacks_list,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        shuffle=True,
                        workers=4)
#     printGraph(history)
    return model
x_train, y_train, x_test=read_data();
x_train, y_train, x_test=prepare_data(x_train, y_train, x_test)
# visualize_data(x_train)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
validation_split=0.0)

model=create_model()
model=run()
prd = model.predict(x_test)
prd_y = np.argmax(prd, axis=1)
save_csv(prd_y)
# model.save('cnn-v10-test.h5')
# print("loading...")
# model = keras.models.load_model(addr+'cnn.h5')
