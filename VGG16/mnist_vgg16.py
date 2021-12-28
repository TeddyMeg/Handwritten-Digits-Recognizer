import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D,Dropout
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import Sequential


from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import numpy as n
import os

def get_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
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

def get_vgg_model():
    model = Sequential()
    #model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=((28, 28, 1)), activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(data_format="channels_last", pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(data_format="channels_last", pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(data_format="channels_last", pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(data_format="channels_last", pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))


    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(lr=0.0001),
        metrics=['accuracy']

    )

    return model

def train_vgg_model(model, x_train, y_train):
    print('Training VGG16 model with batch size 32 and 5 epochs......')
    model.summary()
    hist=model.fit(x_train, y_train, validation_split=0.2,batch_size=32, epochs=5)
    plot_accuracy_and_loss(hist)
    # save_vgg_model(model)
    print("The VGG16 model has successfully trained")


def evaluate_vgg_model(model, x_test, y_test):
  print('Evaluating VGG16 model..........')
  score = model.evaluate(x_test, y_test, batch_size=32, verbose=2)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

def predict_vgg_model(model, x_test, y_test):
  print('Predicting VGG16 model..........')
  predictions = model.predict(x_test, batch_size=32)
  pred = np.argmax(predictions, axis=1)
  print('VGG16 Prediction Score...........')
  print(classification_report(y_test, pred))
  cm = confusion_matrix(y_true=y_test, y_pred=pred)
  cm_plot_labels = [str(i) for i in range(10)]
  plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='VGG16 Confusion Matrix')

def plot_accuracy_and_loss(hist):
  plt.figure(1, figsize=(15, 5))
  plt.subplot(1, 2, 1)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.plot(hist.history["loss"], label="Training Loss")
  plt.plot(hist.history["val_loss"], label="Validation Loss")
  plt.grid(True)
  plt.legend()
  plt.title('VGG16 Training and Validation Loss')
  plt.subplot(1, 2, 2)
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.plot(hist.history["accuracy"], label="Training Accuracy")
  plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
  plt.grid(True)
  plt.legend()
  plt.title('VGG16 Training and Validation Accuracy')
  os.chdir('/..')
  plt.savefig('C:/Users/86155/PycharmProjects/Hand-Written-Digits-Recognizer/plots/VGG16-accuracy-and-loss-plot.png')


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
  plt.savefig('C:/Users/86155/PycharmProjects/Hand-Written-Digits-Recognizer/plots/VGG16-Confusion-Matrix.png')

def save_vgg_model(vgg_model):
  vgg_model.save('C:/Users/86155/PycharmProjects/Hand-Written-Digits-Recognizer/models/MNIST_vgg_16_model.h5')




