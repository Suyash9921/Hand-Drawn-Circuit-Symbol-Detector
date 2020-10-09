import keras

from keras.models import load_model
import cv2
import numpy as np

from keras.applications.vgg16 import VGG16

model = load_model('model2.h5')


from keras.preprocessing.image import load_img
# load an image from file
image = load_img('test1.jpg', target_size=(100, 100))


from keras.preprocessing.image import img_to_array
# convert the image pixels to a numpy array
image = img_to_array(image)

# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

from keras.applications.vgg16 import preprocess_input
# prepare the image for the VGG model
image = preprocess_input(image)

# predict the probability across all output classes
yhat = model.predict(image)