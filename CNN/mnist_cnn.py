# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers,regularizers
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def get_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  input_shape = (28, 28, 1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  print('####################################')
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')
  print('y_train:',y_train)
  print('y_test:', y_test)
  print('####################################')
  return x_train, y_train, x_test, y_test

def get_cnn_model():
  inputs = keras.Input(shape = (28,28,1))
  x = layers.Conv2D(32,3,padding='same',activation='relu',kernel_regularizer=regularizers.l2(0.01))(inputs)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.MaxPool2D()(x)
  x = layers.Conv2D(64,5,padding='same',activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.MaxPool2D()(x)
  x = layers.Conv2D(128,3,padding='same',activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x= layers.MaxPool2D()(x)
  x = layers.Flatten()(x)
  x = layers.Dense(120,activation='relu')(x)
  outputs = layers.Dense(10,activation='softmax')(x)
  model = keras.Model(inputs=inputs,outputs=outputs)
  model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.0001),
    metrics=['accuracy']

  )
  model.summary()
  return model

def train_cnn_model(model, x_train, y_train):
    print('Training CNN model with batch size 32 and 5 epochs......')
    model.summary()
    hist=model.fit(x_train, y_train, validation_split=0.2,batch_size=32, epochs=5)
    plot_accuracy_and_loss(hist)
    # save_cnn_model(model)

    print("The CNN model has successfully trained")


def evaluate_cnn_model(model, x_test, y_test):
  print('Evaluating CNN model..........')
  score = model.evaluate(x_test, y_test, batch_size=32, verbose=2)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])


def predict_cnn_model(model, x_test, y_test):
  print('Predicting CNN model..........')
  predictions = model.predict(x_test, batch_size=32)
  pred = np.argmax(predictions, axis=1)
  print('CNN Prediction Score...........')
  print(classification_report(y_test, pred))
  cm = confusion_matrix(y_true=y_test, y_pred=pred)
  cm_plot_labels = [str(i) for i in range(10)]
  plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='CNN Confusion Matrix')

def plot_accuracy_and_loss(hist):
  plt.figure(1, figsize=(15, 5))
  plt.subplot(1, 2, 1)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.plot(hist.history["loss"], label="Training Loss")
  plt.plot(hist.history["val_loss"], label="Validation Loss")
  plt.grid(True)
  plt.legend()
  plt.title('CNN Training and Validation Loss')
  plt.subplot(1, 2, 2)
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.plot(hist.history["accuracy"], label="Training Accuracy")
  plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
  plt.grid(True)
  plt.legend()
  plt.title('CNN Training and Validation Accuracy')
  plt.savefig('C:/Users/86155/PycharmProjects/Hand-Written-Digits-Recognizer/plots/CNN-accuracy-loss-plot.png')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  plt.figure(2, figsize=(10, 10))
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  #change with your working directory
  plt.savefig('C:/Users/86155/PycharmProjects/Hand-Written-Digits-Recognizer/plots/CNN-Confusion-Matrix.png')


def save_cnn_model(cnn_model):
  cnn_model.save('C:/Users/86155/PycharmProjects/Hand-Written-Digits-Recognizer/models/MNIST_CNN_model.h5')













# See PyCharm help at https://www.jetbrains.com/help/pycharm/
