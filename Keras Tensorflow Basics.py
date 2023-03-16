########################################################################
# In this section we will import the libraries needed to execute the code
########################################################################
# These are the libraries needed to create and process the training dataset and labels
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# These are the libraries needed to create the deep neural network model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

########################################################################
# In this section we will create a dataset along with corresponding labels for training
########################################################################

# First, lets grab some data. We need samples and their corresponding labels.

# Initialize the label and sample arrays

train_labels = []
train_samples = []

# initialize the dataset.
# Example Data: An experimental drug was tested on individuals from ages 13 to 100 in a clinical trial.
# The trial had 2100 participants. Half were under 65 years old, half were 65 or older.
# 95% of patients 65 or older experienced side effects.

for i in range(50):
    # the 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # the 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # the 95% of younger people who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # the 95% of older people who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

# convert the train labels and samples lists into numpy arrays
# then shuffle them to eliminate ordering
# finally, we will scale the training samples down to between 0 and 1

train_samples = np.array(train_samples)
train_labels = np.array(train_labels)
train_samples, train_labels = shuffle(train_samples, train_labels)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

# let's print out the scaled train samples
#for i in scaled_train_samples:
    #print(i)

########################################################################
# Now, we will create the sequential model using Tensorflow
########################################################################

# First, let's check to see if Tensorflow recognizes that we have a GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))


# Now, let's build the model.  Using the Sequential object, tf.keras will build a sequential stack of layers.
# Sequential Object accepts a list, and each element of the list should be a layer

# notice that the input layer is not strictly specified beyond the input_shape argument
# in this example, only the 3 dense layers are defined within the list fed to the Sequential Object

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

# Let's print the model out to see what it looks like

print(model.summary())

########################################################################
# Now, we will train the Neural Network
########################################################################

# Now that the model has been created, we must compile it and specify the optimizer,
# learning rates, and metrics

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Time to train using the fit function. The first parameter, X, is our input data. Y is
# our label data. The batch size is specified next, followed by the number of epochs,
# and the shuffle parameter. Finally, verbose is set to 2 so that we can see the results of each epoch.

# model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, verbose=2)

# Let's now make a validation dataset that will allow us to test our model
# With data that it has not been trained on
# There are two ways to do this using Model.Fit
# One way is to assign the validation data argument to a pre-made validation dataset
# The other is to assign the validation split argument to a decimal between 0 and 1
# This will split the training set into a fraction to be used as the validation set
# Since the validation data is selected before the default shuffling in model.fit,
# We need to make sure that the dataset is shuffled ahead of time (which we did when
# We created the dataset)

model.fit(
      x=scaled_train_samples,
      y=train_labels,
      validation_split=0.1,
      batch_size=10,
      epochs=30,
      verbose=2)

# With this training set up, we will see loss and accuracy metrics
# for the training and validation datasets

