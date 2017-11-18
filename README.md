# Deep learning praticals

## 1. First step in Keras: classifying handwritten digits (MNIST)

#### *The usage of Tensorboard/ ModelCheckpoint for training and saving different models*

1. One input layer + one Dense layer + one output layer(softmax)  
 Corssentropy loss + Adam optimizer  + no normalization   

 ![Deep1](https://github.com/ftZHOU/readmePicture/blob/master/deep_1.png)

2. Same model with normalization (lamda layer)   
 ![Deep2](https://github.com/ftZHOU/readmePicture/blob/master/deep_2.png)

3. Two hidden layers (Dense+relu) + One output layer (softmax)
 ![Deep3](https://github.com/ftZHOU/readmePicture/blob/master/deep_3.png)

4. add regularizer (in the base of the model below)
   1. L2 penality
                x = Dense(256,kernel_regularizer=regularizers.l2(1e-5))(xl)  
    Test loss: 0.158592546591  Test accuracy: 0.9762
   2. Dropout
                x = Dense(hidden1)(x)
                x = Activation('relu')(x)
                x = Dropout(0.5)(x)
    Test loss: 0.0925885819901  Test accuracy: 0.9746                                                                                                                                                                                                                                                                                                



##  Reference link 
[A Friendly Introduction to Cross-Entropy Loss](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/#cross-entropy)  

  