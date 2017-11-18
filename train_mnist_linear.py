import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages')
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Input, Dense, Activation
from keras.models import Model
import os
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import h5py
from keras.models import load_model
from keras.layers import Lambda
from keras import regularizers
from keras.layers import Dropout

# this code run in python 3.5 for tensorflow but my python is 3.6 so manually add a h5py path 

#print(sys.path)


# download and reshape dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_train  = X_train.shape[0]
num_test   = X_test.shape[0]

img_height = X_train.shape[1]
img_width  = X_train.shape[2]
X_train = X_train.reshape((num_train, img_width * img_height))
X_test  = X_test.reshape((num_test, img_width * img_height))
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# building network and normalize data with lambda, using 2 hidden layers and relu function
num_classes = 10
#xi      = Input(shape=(img_height*img_width,))
#xo      = Dense(num_classes)(xi)
#yo      = Activation('softmax')(xo)
#model   = Model(inputs=[xi], outputs=[yo])


xi      = Input(shape=(img_height*img_width,))

mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-5
xl = Lambda(lambda image, mu, std: (image - mu) / std,
           arguments={'mu': mean, 'std': std})(xi)

# L2 regularization
#x = Dense(256,kernel_regularizer=regularizers.l2(1e-5))(xl)  
#x = Activation('relu')(x)
#x = Dense(256,kernel_regularizer=regularizers.l2(1e-5))(x)  
#x = Activation('relu')(x)

# Dropout regularization
x = Dense(256)(xl)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

xo = Dense(num_classes, name="y")(x)

#xo = Dense(num_classes, name="y")(xl) this is when just using one layer
yo = Activation('softmax', name="y_act")(xo)
model = Model(inputs=[xi], outputs=[yo])
#return model

#model.summary()

# prepare logpath for tensorflow http://localhost:6006 
def generate_unique_logpath(logdir, raw_run_name):
        i = 0
        while(True):
                run_name = raw_run_name + "-" + str(i)
                log_path = os.path.join(logdir, run_name)
                if not os.path.isdir(log_path):
                        return log_path
                i = i + 1
run_name = "linear"
#logpath = generate_unique_logpath("./logs_linear", run_name)
#logpath = generate_unique_logpath("./logs_linear1", run_name)
logpath = generate_unique_logpath("./logs_linear3", run_name)

tbcb = TensorBoard(log_dir=logpath)
checkpoint_filepath = os.path.join(logpath,  "best_model.h5")
checkpoint_cb = ModelCheckpoint(checkpoint_filepath, save_best_only=True)



# train the model for 10 seperate time "for iter in $(seq 1 10); do echo ">>>> Run $iter" && python3 train_mnist_linear.py ; done;""
model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_split=0.1,callbacks=[tbcb,checkpoint_cb])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save the best model 

with h5py.File(checkpoint_filepath, 'a') as f:
    if 'optimizer_weights' in f.keys():
        del f['optimizer_weights']

model = load_model(checkpoint_filepath)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
