"""
    Testing code for different neural network configurations.
    Use this code to train, evaluate and save different neural networks

"""
from tensorflow import keras
# ---------------------------------
# - Train, Evaluate And Save ANN:
#----------------------------------
# import ANN.mnist_ann as ann
# x_train, y_train, x_test, y_test = ann.get_data()
# ann_model = ann.get_ann_model()
# # keras.utils.plot_model(
# #     ann_model,
# #     to_file="plots/ann_model.png",
# #     show_shapes=False,
# #     show_dtype=False,
# #     show_layer_names=True,
# #     rankdir="TB",
# #     expand_nested=False,
# #     dpi=96,
# #     layer_range=None,
# # )
# ann.train_ann_model(ann_model,x_train, y_train)
# ann.evaluate_ann_model(ann_model,x_test, y_test)
# ann.predict_ann_model(ann_model,x_test,y_test)


# ---------------------------------
# - Train, Evaluate And Save CNN:
#----------------------------------
# import CNN.mnist_cnn as cnn
# x_train, y_train, x_test, y_test = cnn.get_data()
# cnn_model = cnn.get_cnn_model()
# # keras.utils.plot_model(
# #     cnn_model,
# #     to_file="plots/cnn_model.png",
# #     show_shapes=False,
# #     show_dtype=False,
# #     show_layer_names=True,
# #     rankdir="TB",
# #     expand_nested=False,
# #     dpi=96,
# #     layer_range=None,
# # )
# cnn.train_cnn_model(cnn_model,x_train, y_train)
# cnn.evaluate_cnn_model(cnn_model,x_test, y_test)
# cnn.predict_cnn_model(cnn_model,x_test,y_test)


# ---------------------------------
# - Train, Evaluate And Save VGG16:
#----------------------------------
import VGG16.mnist_vgg16 as vgg16
x_train, y_train, x_test, y_test = vgg16.get_data()
vgg_model = vgg16.get_vgg_model()
# keras.utils.plot_model(
#     vgg_model,
#     to_file="plots/vgg_model.png",
#     show_shapes=False,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
#     layer_range=None,
# )
vgg16.train_vgg_model(vgg_model,x_train, y_train)
# vgg16.evaluate_vgg_model(vgg_model,x_test, y_test)
# vgg16.predict_vgg_model(vgg_model,x_test,y_test)






