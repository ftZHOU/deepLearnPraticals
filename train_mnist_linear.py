from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Input, Dense, Activation
from keras.models import Model
import os
from keras.callbacks import TensorBoard

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 10 handwritting numbers as input 
num_classes=10

# transfer input as a vector to use keras
num_train=X_train.shape[0]
num_test=X_test.shape[0]
img_height=X_test.shape[1]
img_width=X_test.shape[2]
X_train = X_train.reshape((num_train,img_height*img_width))
X_test = X_test.reshape((num_test, img_width * img_height))

# transfer label to an one-hot coding
y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test,num_classes)

# build the network using the soft-max
xi= Input(shape=(img_height*img_width,))
xo=Dense(num_classes)(xi)
yo=Activation('softmax')(xo)
model=Model(inputs=[xi],outputs=[yo])
model.summary()

# compile and train
model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_split=0.1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# define a unique path function to log data
def generate_unique_logpath(logdir, raw_run_name):
        i = 0
        while(True):
                run_name = raw_run_name + "-" + str(i)
                log_path = os.path.join(logdir, run_name)
                if not os.path.isdir(log_path):
                        return log_path
                i = i + 1

run_name = "linear"
logpath = generate_unique_logpath("./logs_linear", run_name)
tbcb = TensorBoard(log_dir=logpath)
model.fit(X_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_split=0.1,
          callbacks=[tbcb])