# Handwritten-Digits-Recognizer
The objective of this project is to build an image-classifier using Neural Networks to accurately categorize the handwritten digits. It also has a gui made using tkinter and OpenCV where a user can draw a digit and choose a model to predict what the number is.

**Network Architecture**

I used three types of neural networks, to build the models and evaluate them to compare their performance.
These are:-
1. ANN (Artificial Neural Network)
2. CNN (Convolutional Neural Network).
3. VGG-16 (Deep Neural Network)

**Training, Validation And Test Sets**

For the purpose of this experiment the dataset was split into 3 categories.
1. 48,000 Training Sets
2. 12,000 Validation Sets
3. 10,000 Test Sets

**Environment Setup**

I recommend using Anaconda for running the script.

Run the following command on conda prompt to create a new environment with all the required packages.
conda env create -f mnist_env_packages.yml

The mnist_env_packages.yml file can be found in the repo, which contains details of all the required packages to run the script.

**Input Dimension**

Each image is 28x28 pixels with 1 channel.

**Project Structure**

1. The pre-trained models can be found in /models/ directory
2. All the plots and charts for the training and evaluation can be found in /plots/ folder
3. test.py is the file used to run the code for training and evaluating the three models.
4. gui_digit_recognizer.py contains the code to run the gui for predicting handwritten digits
5. The three sub folders /ANN/, /CNN/, and /VGG16/ contains the python files which contains the code for building the three models respectively

**Results
Training and Validation Accuracy Plot**

![ANN-accuracy-loss-plot](https://user-images.githubusercontent.com/37738265/147532019-d9ee91e7-5659-4f1b-a7ff-ad13f23f861e.png)
![CNN-accuracy-loss-plot](https://user-images.githubusercontent.com/37738265/147532022-43839bd1-0798-48bb-8c9e-bb80a5e58139.png)
![VGG16-accuracy-and-loss-plot](https://user-images.githubusercontent.com/37738265/147532026-9fccced7-a0f5-41cc-b028-2a3d03694c8c.png)

**Confusion Matrix**

![ANN-Confusion-Matrix](https://user-images.githubusercontent.com/37738265/147532057-adba68b2-f596-4afa-aa9b-92a7d7d0ec09.png)
![CNN-Confusion-Matrix](https://user-images.githubusercontent.com/37738265/147532060-20c19b1f-36c4-4b19-bcc9-00c2771fd44d.png)
![VGG16-Confusion-Matrix](https://user-images.githubusercontent.com/37738265/147532065-1b54ed9f-8e33-462d-b155-c2702a03cdc3.png)


