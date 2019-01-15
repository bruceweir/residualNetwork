
# coding: utf-8

# In[43]:


from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Reshape, Dropout, Flatten, BatchNormalization, Input
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.activations import *
from tensorflow.python.keras.utils import *
from tensorflow.python.keras.losses import *
from tensorflow.python.keras.optimizers import *

import matplotlib.pyplot as plt
import numpy as np


# In[29]:


def ResidualUnit(layer_in):
    
    residual = layer_in
    
    out = BatchNormalization()(layer_in)
    out = Activation("relu")(out)
    out = Conv2D(filters=32, kernel_size=[3, 3], padding="same")(out)
    
    out = keras.layers.add([residual, out])
    


# In[50]:


img_rows = 28
img_cols = 28
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[51]:


plt.imshow(X_train[0].reshape(28, 28))
plt.show()


# In[59]:


model = Sequential()

model.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=[3, 3], padding="same"))
model.add(Conv2D(filters=8, kernel_size=[3, 3], activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])



# In[63]:


num_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])


# In[58]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[64]:


batch_size = 128
epochs = 12

model.fit(x_train, y_train, 
         batch_size=batch_size,
         epochs=epochs,
         verbose=1,
         validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)


# In[ ]:


print('Test loss:', score[0])
print('Test accuracy:', score[1])

