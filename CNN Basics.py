########################################################################
# This file will show the basics of how to utilize a CNN to categorize images
########################################################################

#Import Section
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt

########################################################################
# Section where we make functions we will need later
########################################################################
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#Download the dataset from Kaggle at this link: https://www.kaggle.com/c/dogs-vs-cats/data

# Organize data into train, valid, test dirs
# This code looks into the data directory and organizes the mass of data into training, test, and validation sets
os.chdir('Data/Dogs and Cats/train')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for i in random.sample(glob.glob('cat*'), 500):
        shutil.move(i, 'train/cat')
    for i in random.sample(glob.glob('dog*'), 500):
        shutil.move(i, 'train/dog')
    for i in random.sample(glob.glob('cat*'), 100):
        shutil.move(i, 'valid/cat')
    for i in random.sample(glob.glob('dog*'), 100):
        shutil.move(i, 'valid/dog')
    for i in random.sample(glob.glob('cat*'), 50):
        shutil.move(i, 'test/cat')
    for i in random.sample(glob.glob('dog*'), 50):
        shutil.move(i, 'test/dog')

os.chdir('../../')

# If the labels for the test data were not provided/unknown, then the directory structure would look like
# test\unknown\ for the test directory structure

########################################################################
# This section will cover processing the data
########################################################################

# Check for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# We then create variables for which the paths to the train, valid, and test data directories are assigned.

train_path = r'C:\Users\brmar\PycharmProjects\Deep-Learning-\Data\Dogs and Cats\train\train'
valid_path = r'C:\Users\brmar\PycharmProjects\Deep-Learning-\Data\Dogs and Cats\train\valid'
test_path = r'C:\Users\brmar\PycharmProjects\Deep-Learning-\Data\Dogs and Cats\train\test'

# Now, we use Keras' ImageDataGenerator class to create batches of data from the train, valid, and test directories.

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

'''
ImageDataGenerator.flow_from_directory() creates a DirectoryIterator, which
generates batches of normalized tensor image data from the respective data directories.
Notice, to ImageDataGenerator for each of the data sets, we specify 
preprocessing_function=tf.keras.applications.vgg16.preprocess_input
This is an image preprocessing function designed to be used specifically for the vgg16 CNN, which we will be 
using for this dataset
If we were making our own model from scratch, we would need to make our own preprocessing function.

To flow_from_directory(), we first specify the path for the data. We then specify the target_size of the images, which 
will resize all images to the specified size in pixels. The size we specify here is determined by 
the input size that the neural network expects.

The classes parameter expects a list that contains the underlying class names, and lastly, we specify the batch_size.
We also specify shuffle=False only for test_batches. That's because, later when we plot the evaluation results from the 
model to a confusion matrix, we'll need to able to access the unshuffled labels for the test set. 
By default, the data sets are shuffled.

Note, in the case where you do not know the labels for the test data, you will need to modify the test_batches variable. 
Specifically, the change will be to set the parameters classes = None and class_mode = None in flow_from_directory().
'''

# Visualize the data using the iterator we just created

imgs, labels = next(train_batches)
plotImages(imgs)
print(labels)

# The images shown will have gone through the preprocessing step, so the rgb data has been adjusted for model input.
# Note that dogs are represented with the one-hot encoding of [0,1], and cats are represented by [1,0].

########################################################################
# This section will cover building and training the CNN
########################################################################

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=10,
    verbose=2
)

'''
We need to specify steps_per_epoch to indicate how many batches of samples from our training 
set should be passed to the model before declaring one epoch complete. Since we have 1000 samples in our training set, 
and our batch size is 10, then we set steps_per_epoch to be 100, since 100 batches of 10 
samples each will encompass our entire training set.

We're able to use len(train_batches) as a more general way to specify this value, 
as the length of train_batches is equal to 100 since it is made up of 100 batches of 10 samples. 
Similarly, we specify validation_steps in the same fashion but with using valid_batches.

We're specifying 10 as the number of epochs we'd like to run, and setting the verbose parameter to 2, 
which just specifies the verbosity of the log output printed to the console during training.

Notice that the model overfits to the training data. This means that we will need to either:
1) Work with this current model to reduce overfitting.
2) Load a pre-trained model and fine-tune it to this dataset. 

'''

