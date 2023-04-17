
'''
This file will introduce MobileNets, a class of DCNNs that are
Lightweight in comparison to other models.
MobileNets are vastly smaller in memory size and faster in performance
than many other models.

For example, VGG16 comprises 138,000,00 parameters and takes up 553 MB of disk space
One of the largest MobileNets takes up 17 MB of disk and has 4,200,000 parameters
This allows the deployment of MobileNets in the browser or on a small device

Here is a link to the MobileNets paper: https://arxiv.org/pdf/1704.04861.pdf

'''
########################################################################
# Imports
########################################################################

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from IPython.display import Image
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

########################################################################
# Methods
########################################################################

'''
Next, we have a function called prepare_image() that accepts an image file, 
and processes the image to get it in a format that the model expects. 
We'll be passing each of our images to this function before we use MobileNet to 
predict on it, so let's see what exactly this function is doing.

Within this function, we first define the relative path to the images.

then call the Keras function image.load_img() which accepts the image file and a 
target_size for the image, which we're setting to (224,224) 
(which is the default size for MobileNet). 
load_img() returns an instance of a PIL image.

then convert the PIL image into an array with the Keras img_to_array() function, 
and then we expand the dimensions of that array by using numpy's expand_dims()

finally, call preprocess_input() from tf.keras.applications.mobilenet, which preprocesses the 
given image data to be in the same format as the images that MobileNet was 
originally trained on. Specifically, it's scaling the pixel values in the 
image between -1 and 1, and this function will return the preprocessed image data 
as a numpy array.

'''
def prepare_image(file):
    img_path = 'Data/MobileNet/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

########################################################################
# Import and save the model
########################################################################

mobile = tf.keras.applications.mobilenet.MobileNet()
print(mobile.summary())

# This will save the model to disk
mobile.save('models/MobileNet.h5')

# Use the load model method when you want to use this model from disk instead of downloading it again
#mobile = load_model('MobileNet.h5')
#print(mobile.summary())

########################################################################
# Import sample images and perform predicitons on them with MobileNet
########################################################################

Image(filename='Data/MobileNet/lizard.png', width=300,height=200)

'''
First, we are going to process this image by passing it to our prepare_image() function and
assign the result to this preprocessed_image variable. 
We're then having MobileNet predict on this image by calling mobile.predict() 
and passing it our preprocessed_image.'''

preprocessed_image = prepare_image('lizard.png')
predictions = mobile.predict(preprocessed_image)

results = imagenet_utils.decode_predictions(predictions)
print(results)

'''
The image was of a lizard that was green, and the model gave that class the highest probability
so the model did well at assigning that class the highest probability. 
The remaining four classes are all different types of similar lizards as well,
so overall the model did a good job at classifying this one.
'''

# let's try a picture of a cup of espresso

Image(filename='Data/MobileNet/Espresso.png', width=300,height=200)
preprocessed_image = prepare_image('Espresso.png')
predictions = mobile.predict(preprocessed_image)

results = imagenet_utils.decode_predictions(predictions)
print(results)

# Finally, let's try an image of a strawberry

Image(filename='Data/MobileNet/strawberry.png', width=300,height=200)
preprocessed_image = prepare_image('strawberry.png')
predictions = mobile.predict(preprocessed_image)

results = imagenet_utils.decode_predictions(predictions)
print(results)

'''
MobileNet correctly classified the espresso and strawberry images
With the correct class given over 99 percent probability. 
Very good!
'''