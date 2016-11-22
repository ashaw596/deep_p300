'''Transfer learning toy example:
1- Train a simple convnet on the MNIST dataset the first 5 digits [0..4].
2- Freeze convolutional layers and fine-tune dense layers
   for the classification of digits [5..9].
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_transfer_cnn.py
Get to 99.8% test accuracy after 5 epochs
for the first five digits classifier
and 99.2% for the last five digits after transfer + fine-tuning.
'''

from __future__ import print_function
import numpy as np
import datetime
import scipy.io as sio
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from keras.utils.np_utils import convert_kernel

from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing

import subprocess
import shutil
import uuid

import sys
import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
      '--output_path',
      type=str,
      help='The path to which checkpoints and other outputs '
      'should be saved. This can be either a local or GCS '
      'path.')
parser.add_argument(
      '--data_path',
      type=str,
      help='The path of the data.')
    
args, _ = parser.parse_known_args()

np.random.seed(1337)  # for reproducibility

env = json.loads(os.environ.get('TF_CONFIG', '{}'))


if True:
    import tensorflow as tf
    server = tf.train.Server.create_local_server()
    sess = tf.Session(server.target)
    K.set_session(sess)

if  args.data_path.startswith('gs://'):
    data_path = os.path.join('/tmp/', str(uuid.uuid4()))
    os.makedirs(data_path)
    subprocess.check_call(['gsutil', '-m', '-q', 'cp', '-r', os.path.join(args.data_path, '*'), data_path])
else:
    data_path = args.data_path
    args.data_path = None

if  args.output_path.startswith('gs://'):
    output_path = os.path.join('/tmp/', str(uuid.uuid4()))
    os.makedirs(output_path)
else:
    output_path = args.output_path
    args.output_path = None

matlabFile = os.path.join(data_path, 'data.mat')
matlab = sio.loadmat(matlabFile, squeeze_me=True, struct_as_record=False)

data = matlab['data']

X = data.X
Y = data.Y
subject = data.subject
session = data.session

print(X.shape)
print(Y.shape)
print(subject.shape)
print(session.shape)
# Cz, Pz, Oz
channels = [31, 12, 15];
X = X[:, channels,:]
#X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
#X = preprocessing.scale(X)
#X = X.reshape((X.shape[0], len(channels), X.shape[1]/len(channels)))



#print(sum(Y))
Y[Y==-1] = 0 
#print(sum(Y))
#print(1)
#print(X.shape)
#print( )
#print(training)
#print(training.shape)
testX = X[subject>6, :, :]
testY = Y[subject>6]
trainX = X[subject<=6, :, :]
trainY = Y[subject<=6]

#print (testX.shape)
#print (trainX.shape)


print(testX.shape)
print(trainX.shape)
#input_shape = (64, 3)

batch_size = 64
nb_classes = 2
nb_epoch = 300
img_rows, img_cols = 3, 64

if K.image_dim_ordering() == 'th':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

# input image dimensions
# number of convolutional filters to use
nb_filters = 3
# size of pooling area for max pooling
pool_size = 2
# convolution kernel size
kernel_size_rows = 1
kernel_size_cols = 3

print (input_shape)
# define two groups of layers: feature (convolutions) and classification (dense)
feature_layers = [
    Convolution2D(32, 1, 3,
                  border_mode='valid',
                  input_shape=input_shape),
    Activation('sigmoid'),
    Convolution2D(32, 1, 3,
                  border_mode='valid',
                  input_shape=input_shape),
    Activation('relu'),
    MaxPooling2D(pool_size=(1, 2)),
    Convolution2D(64, 3, 3,
                  border_mode='valid',
                  input_shape=input_shape),
    Activation('relu'),
    MaxPooling2D(pool_size=(1, 2)),
    Dropout(0.25),
    Flatten(),
]
classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(nb_classes),
    Activation('softmax')
]

# create complete model
model = Sequential(feature_layers + classification_layers)

model.summary()
model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])


def train_model(model, train, test, nb_classes):
    print((train[0].shape[0],) + input_shape)
    X_train = train[0].reshape((train[0].shape[0],) + input_shape)
    X_test = test[0].reshape((test[0].shape[0],) + input_shape)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    #X_train /= 255
    #X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(train[1], nb_classes)
    Y_test = np_utils.to_categorical(test[1], nb_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    t = now()
    hist = model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1,
              validation_data=(X_test, Y_test))
    print(hist.history)

    print('Training time: %s' % (now() - t))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    Y_pred = model.predict(X_test)
    print(Y_pred)
    y_pred = np.argmax(Y_pred, axis=1)
    print(y_pred)
    target_names = ['class 0(Not)', 'class 1(P300)']
    print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
    print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

def test_model(model, test, nb_classes):
    X_test = test[0].reshape((test[0].shape[0],) + input_shape)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np_utils.to_categorical(test[1], nb_classes)
    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


now = datetime.datetime.now

# train model for 5-digit classification [0..4]
train_model(model,
            (trainX, trainY),
            (testX, testY), nb_classes)

task_data = env.get('task', None) or {'type': 'master', 'index': 0}
task = type('TaskSpec', (object,), task_data)

if (not task or task.type == 'master'):
    out = os.path.join(output_path, 'p300_model.h5')
    print(out)
    print('save')
    model.save(out)
    model.save_weights(os.path.join(output_path, 'p300_weights.h5'))

if  args.data_path:
    shutil.rmtree(data_path, ignore_errors=True)

if  args.output_path:
    subprocess.check_call(['gsutil', '-m', '-q', 'cp', '-r', os.path.join(output_path, '*'), args.output_path])
    shutil.rmtree(output_path, ignore_errors=True)


                  



"""
# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False

# transfer: train dense layers for new classification task [5..9]
train_model(model,
            (X_train_gte5, y_train_gte5),
            (X_test_gte5, y_test_gte5), nb_classes)
"""