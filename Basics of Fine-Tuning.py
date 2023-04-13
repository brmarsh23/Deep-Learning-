########################################################################
# This file will show the basics of how to utilize a pre-trained CNN for fine-tuning
# This is a companion file to the "CNN Basics File"
# The VGG16 paper is linked here: https://arxiv.org/pdf/1409.1556.pdf
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
from tensorflow.keras.models import load_model
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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

########################################################################
# This Section remakes the dataset in the event that the dataset has been removed
########################################################################
'''
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

'''
# Let's remake the datasets from the CNN Basics File. In the future, we will learn how to import these variables
# From the CNN Basics File

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



imgs, labels = next(train_batches)
plotImages(imgs)
print(labels)
########################################################################
# Now we will import the VGG16 model. This piece of code requires an internet connection
# Once the model is saved to the computer, code is in place that will load the model from disk
########################################################################

#vgg16_model = tf.keras.applications.vgg16.VGG16()
#print(vgg16_model.summary())

# This will save the model to disk
#vgg16_model.save('models/vgg16.h5')

# Use the load model method when you want to use this model from disk instead of downloading it again
vgg16_model = load_model('models/vgg16.h5')
print(vgg16_model.summary())

'''
VGG16 is much more complex and sophisticated and has many more layers than the CNN Basics Model.
Notice that the last Dense layer of VGG16 has 1000 outputs. 
These outputs correspond to the 1000 categories in the ImageNet library.

Since we're only going to be classifying two categories, cats and dogs, 
we need to modify this model in order for it to do what we want it to do, which is to only classify cats and dogs.

Before we do that, note that the type of Keras models we've been working with so far in this series 
have been of type Sequential.

If we check out the type of model vgg16_model is, we see that it is of type Model, 
which is from the Keras' Functional API.
'''

print(type(vgg16_model))

########################################################################
# For now, we're going to go through a process to convert the Functional model to a Sequential model,
# so that it will be easier for us to work with given our current knowledge.
# We first create a new model of type Sequential.
# We then iterate over each of the layers in vgg16_model, except for the last layer,
# and add each layer to the new Sequential model.
########################################################################

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

'''
Now, we have replicated the entire vgg16_model (excluding the output layer) to a new Sequential model, 
which we've just given the name model.

Next, we'll iterate over each of the layers in our new Sequential model and set them to be non-trainable. 
This freezes the weights and other trainable parameters in each layer so that they will not be trained or updated when
we later pass in our images of cats and dogs.
'''

for layer in model.layers:
    layer.trainable = False

'''
Since VGG16 is already really good at classifying images,
We only want to modify the model such that the output layer understands only how to classify cats and dogs 
and nothing else. Therefore, we don't want any re-training to occur on the earlier layers.

Next, we add our new output layer, consisting of only 2 nodes that correspond to cat and dog. 
This output layer will be the only trainable layer in the model.

'''

model.add(Dense(units=2, activation='softmax'))
print(model.summary())

########################################################################
# Time to Compile and then train the model
########################################################################

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

'''
Similar to how we've compiled models in previous files, 
we'll use the Adam optimizer with a learning rate of 0.0001, 
categorical_crossentropy as our loss, and ‘accuracy' as our metric.
'''

model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=5,
          verbose=2
)

# Only 5 epochs of training this time
# The most noticeable improvement is that this model is generalizing very well to the validation data,
# unlike the CNN we built from scratch in the CNN basics file. This is because the vgg16 model was originally
# trained with images of cats and dogs in distribution (in the original dataset).
# all we had to do was fine-tune the last layer to recognize only cats or dogs.

########################################################################
# Now we will predict with the fine-tuned model
########################################################################

# let's look at the test images. They should be preprocessed in the same way as the training and validation sets
# Remember, do not shuffle the test images. You want the test images and the test labels to match up.

test_imgs, test_labels = next(test_batches)
plotImages(test_imgs)
print(test_labels)

# Time to predict

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

# Plot the confusion matrix

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels = ['cat','dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')