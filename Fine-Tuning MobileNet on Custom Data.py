
'''
This file will apply the concepts of Fine-Tuning to the MobileNet model
using a custom dataset obtained here (greyscale): https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset
and here (RGB): https://github.com/ardamavi/Sign-Language-Digits-Dataset

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
# Create Methods
########################################################################

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
# Organize the RGB Data into Directories and Datasets
########################################################################

'''
First, we create train, valid, and test directories.
We then move all class directories (0 - 9) with their respective images from Sign_Language_Digits_Dataset
into the train directory.
We then make class directories (0 - 9) for the valid and test data sets as well.
We then loop through each class directory in the train directory and randomly move 30 images from each class into 
the corresponding class directory in valid and 5 images from each class into the corresponding class directory in test.

Finally, we end by moving back into the current directory (Not necessary if not using a Jupyter Notebook).
'''

# Once this code has been run on the downloaded dataset, it can be commented out

os.chdir('Data/Sign_Language_Digits_Dataset_RGB')
if os.path.isdir('train/0/') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

    for i in range(0, 10):
        shutil.move(f'{i}', 'train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')

        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for k in test_samples:
            shutil.move(f'train/{i}/{k}', f'test/{i}')
os.chdir('../..')

'''
After our image data is all organized on disk, we need to create the directory iterators for the train, validation,
and test sets in the exact same way as we did for the cat and dog data set in the CNN Basics and Fine-Tuning Basics
Files.

'''


train_path = 'Data/Sign_Language_Digits_Dataset_RGB/train'
valid_path = 'Data/Sign_Language_Digits_Dataset_RGB/valid'
test_path = 'Data/Sign_Language_Digits_Dataset_RGB/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

'''
Notice the preprocessing_function parameter we're supplying to ImageDataGenerator. 
We're setting this equal to tf.keras.applications.mobilenet.preprocess_input. 
This is going to do the necessary MobileNet preprocessing on the images obtained from flow_from_directory().

To review, preprocessing images for MobileNet requires scaling the pixel values in the image between -1 and 1, and 
returning the preprocessed image data as a numpy array.

To flow_from directory(), we're passing in the path to the data set, the target_size to resize the images to, 
and the batch_size we're choosing to use for training. We do this exact same thing for all three data sets.

For test_batches, we're also supplying one additional parameter, shuffle=False, 
which causes the test dataset to not be shuffled, so that we can access the corresponding non-shuffled test labels 
to plot to a confusion matrix later.

'''

########################################################################
# Load the MobileNet model from Internet or Disk and Adjust for Fine-Tuning
########################################################################

# mobile = tf.keras.applications.mobilenet.MobileNet()
# print(mobile.summary())

# This will save the model to disk
# mobile.save('models/MobileNet.h5')

# Use the load model method when you want to use this model from disk instead of downloading it again
mobile = load_model('models/MobileNet.h5')
print(mobile.summary())

# Next, we're going to grab the output from the fifth to last layer of the model
# and store it in variable x.

x = mobile.layers[-5].output

'''
We'll be using this to build a new model. 
This new model will consist of the original MobileNet up to the fifth to last layer. 
We're not including the last four layers of the original MobileNet.

By looking at the summary of the original model, 
we can see that by not including the last four layers, we'll be including everything up to 
and including the last global_average_pooling layer.

Note that the amount of layers that we choose to cut off when you're fine-tuning a model will vary for 
each application. In this task, removing the last 4 layers works well. So with this setup, 
we'll be keeping the vast majority of the original MobileNet architecture, which has 88 layers total.

Now, we need to reshape our output from the global_average_pooling layer that we will pass to our 
output layer, which we're calling output. We use the tf.keras.layers Reshape method to do this.
This method adds a reshape layer that takes the input of the previous layer (in this case None,1,1,1024)
and outputs a (None, 1024) layer. 

The output layer will just be a Dense layer with 10 output nodes for the ten corresponding classes, 
and we'll use the softmax activation function.
'''

x = tf.keras.layers.Reshape(target_shape=(1024,))(x)
output = Dense(units=10, activation='softmax')(x)

# Now, we construct the new fine-tuned model, which we're calling fine_tune_Mobilenet.

fine_tune_Mobilenet = Model(inputs=mobile.input, outputs=output)

'''
Note, you can see by the Model constructor used to create our model, that this is a model 
that is being created with the Keras Functional API, not the Sequential API that we've worked with 
in previous files. That's why this format that we're using to create the model may look a little 
different than what you're used to.

To build the new model, we create an instance of the Model class and specify the inputs to the model to be 
equal to the input of the original MobileNet, and then we define the outputs of the model to be equal to 
the output variable we created directly above.

This creates a new model, which is identical to the original MobileNet up to the original model's sixth to 
last layer. We don't have the last five original MobileNet layers included, but instead we have a new layer,
the output layer we created with ten output nodes.

Now, we need to choose how many layers we actually want to be trained when we train on our new data set.

We still want to keep a lot of what the original MobileNet has already learned from ImageNet by freezing 
the weights in many of layers, especially earlier ones, but we do indeed need to train some layers since 
the model still needs to learn features about this new data set.

I did a little experimenting and found that training the last 22 layers will give us a pretty 
decently performing model, so let's go with that. Feel free to play around with this to see how
freezing and unfreezing layers can change model performance!

So the twenty-third-to-last layer and all layers after it will be trained when we fit the model on 
the new data set. All layers above will not be trained, so their original ImageNet weights 
will stay in place.
'''

for layer in fine_tune_Mobilenet.layers[:-22]:
    layer.trainable = False

print(fine_tune_Mobilenet.summary())

# Compile the Model

fine_tune_Mobilenet.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model

fine_tune_Mobilenet.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=10,
            verbose=2
)

########################################################################
# Perform Inference and Check Results
########################################################################

# Get test labels by grabbing the classes from the unshuffled test set.

test_labels = test_batches.classes

# Run predictions

predictions = fine_tune_Mobilenet.predict(x=test_batches, steps=len(test_batches), verbose=0)

# Create the Confusion Matrix

cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))

'''
Now we are printing the class_indices from our test_batches so that we can see the order of the classes and 
specify them in that same order when we create the labels for our confusion matrix.
If you know the class indices already, no need to do this
'''
print(test_batches.class_indices)

cm_plot_labels = ['0','1','2','3','4','5','6','7','8','9']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# finaL step, save the model for later!

fine_tune_Mobilenet.save('models/Fine_Tune_SL_RGB_MobileNet.h5')
