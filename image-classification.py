# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1 - common preprocessing step that helps with future calculation
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model architecture
model = models.Sequential() #composed of linear stak of layers, functional is harder
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#convolutional layers automatically and adaptively learn different spacial hierarchies  of features
#theyre feature detectors. go around image to find features
model.add(layers.MaxPooling2D((2, 2)))
#helps reduce spacial dimensions of output value to make easier to spot features
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add Dense layers on top
#fully connected so every neuron connected to every other neuron
# decision makers
#take output from convolutional layers and features that conv layer has given, and makes a decision on what image actually is
model.add(layers.Flatten())
# flatten converts 2d matrix of features provided by conv layer into a vector
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(11))
#when change Received a label value of 9 which is outside the valid range of [0, 7).  Label values: 1 1 6 3 2 3 4 1 0 9 9 7 9 4 8 2 6 5 0 1 4 2 8 4 4 2 3 2 5 8 2 8
#has to be int cant be float

# Compile and train the model
model.compile(optimizer='adam', # algorith used to minimise cost
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),#method used to calc difference between prediction and real
              metrics=['accuracy'])#measures performance - in proportion to number of correct predictions

history = model.fit(train_images, train_labels, epochs=3, 
                    validation_data=(test_images, test_labels))
#train image and label are training data
# number of epochs is number of iterations over whole data set
#test images and labels are validation data


# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)