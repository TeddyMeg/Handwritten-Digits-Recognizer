# %load mnist_ann.py
import itertools

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import os
from sklearn.metrics import classification_report,confusion_matrix

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

def get_ann_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28,28,1)),
        layers.Dense(512,activation="relu"),
        layers.Dense(128,activation="relu"),
        layers.Dense(10,activation='softmax')

    ])

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(lr=0.0001),
        metrics=['accuracy']

    )
    model.summary()
    return model


def train_ann_model(model,x_train,y_train):
    print('Training ANN model with batch size 32 and 5 epochs......')
    hist=model.fit(x_train, y_train, validation_split=0.2,batch_size=32, epochs=5)
    plot_accuracy_and_loss(hist)
    # save_ann_model(model)
    print("The ANN model has successfully trained")


def evaluate_ann_model(model,x_test,y_test):
    print('Evaluating ANN model..........')
    score = model.evaluate(x_test, y_test, batch_size=32, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def predict_ann_model(model,x_test,y_test):
    print('Predicting ANN model..........')
    predictions=model.predict(x_test,batch_size=32)
    pred=np.argmax(predictions,axis=1)
    print('ANN Prediction Score...........')
    print(classification_report(y_test,pred))
    cm = confusion_matrix(y_true=y_test, y_pred=pred)
    cm_plot_labels=[str(i) for i in range(10)]
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='ANN Confusion Matrix')

def plot_accuracy_and_loss(hist):
    plt.figure(1, figsize = (15, 5))
    plt.subplot(1,2,1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(hist.history["loss"], label = "Training Loss")
    plt.plot(hist.history["val_loss"], label = "Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.title('ANN Training and Validation Loss')
    plt.subplot(1,2,2)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot( hist.history["accuracy"], label = "Training Accuracy")
    plt.plot( hist.history["val_accuracy"], label = "Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.title('ANN Training and Validation Accuracy')
    os.chdir('/..')
    #change with your working directory
    plt.savefig('C:/Users/86155/PycharmProjects/Hand-Written-Digits-Recognizer/plots/ANN-accuracy-loss-plot.png')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(2, figsize=(10,10))
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
    os.chdir('/..')
    plt.savefig('C:/Users/86155/PycharmProjects/Hand-Written-Digits-Recognizer/plots/ANN-Confusion-Matrix.png')


def save_ann_model(ann_model):
    ann_model.save('C:/Users/86155/PycharmProjects/Hand-Written-Digits-Recognizer/models/MNIST_ANN_model.h5')

