#importing packages

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import os

#loading data
os.chdir("G:\\R work\\Digit recogniser")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv") 

train.head()
test.head()

#Seperating out target variable
y_train = train["label"]
x_train = train.drop(labels = ["label"],axis = 1) 

# visualize number of digits classes
plt.figure(figsize=(15,7))
g = sns.countplot(y_train, palette="icefire")
plt.title("Number of digit classes")
y_train.value_counts() 

# plot some samples
img = x_train.iloc[0].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()

#Normalize the data
x_train = x_train / 255.0
test = test / 255.0
print("x_train shape: ",x_train.shape)
print("test shape: ",test.shape)

#Reshape the data
x_train = x_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1,28,28,1) 

#Label Encoding target variable
y_train = to_categorical(y_train, num_classes = 10)

# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=2)

print("x_train shape",x_train.shape)
print("x_test shape",x_val.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_val.shape)

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()


model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(.025))

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size = (2,2), strides = (2,2) ))

model.add(Dropout(0.25))

#Fully connected layer
model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax")) 

#Optimizer

optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.9)

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ['accuracy'])

#Number of epochs
epochs = 10
batch_size = 250

datagen = ImageDataGenerator(featurewise_center = False, samplewise_center = False, featurewise_std_normalization = False, samplewise_std_normalization = False, zca_whitening = False, rotation_range = 0.5, zoom_range = 0.5, width_shift_range = 0.5, height_shift_range = 0.5, horizontal_flip = False, vertical_flip = False)

datagen.fit(x_train)

#Model fitting

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), epochs = epochs, validation_data= (x_val, y_val) ,steps_per_epoch=x_train.shape[0]//batch_size)

import seaborn as sns

y_pred = model.predict(x_val)

y_pred_classes = np.argmax(y_pred,axis = 1)

y_true = np.argmax(y_val,axis = 1) 

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()




