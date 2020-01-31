# Experiment logbook for FI-2010

## In progress


## Completed

In reverse chronological order of completion time:

* **2020-01-31** on colab  

FI_2010_cnn_LSTM_colab.ipynb

```
model = Sequential()
model.add(Conv2D(16, kernel_size=(4, 40),strides=(1, 1),activation='relu')
model.add(Conv2D(16, kernel_size=(1,1),strides=(1,1),activation='relu'))
model.add(Conv2D(32,kernel_size=(4,1),strides=(1,1),data_format='channels_last',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1),strides=2))
model.add(Conv2D(32,kernel_size=(3,1),strides=(1,1),data_format='channels_last',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1),strides=2))
model.add(Lambda(squeeze_axis))
model.add(LSTM(100,kernel_regularizer=regularizers.l2(0.01),return_sequences=False,activation='relu'))
model.add(Dropout(0.60))
model.add(Dense(3, activation='softmax'))
```

epochs = 100  
F1 test: 0.51    


* **2020-01-30** on colab  

FI_2010_cnn3_colab.ipynb

```
model = Sequential()
model.add(Conv2D(16, kernel_size=(1, 2),strides=(1,2),activation='relu')
model.add(Conv2D(16, kernel_size=(4,1),strides=(1,1),activation='relu'))
model.add(Conv2D(16, kernel_size=(4,1),strides=(1,1),activation='relu'))
model.add(Conv2D(16, kernel_size=(1,2),strides=(1,2),activation='relu'))
model.add(Conv2D(16, kernel_size=(4,1),strides=(1,1),activation='relu'))
model.add(Conv2D(16, kernel_size=(4,1),strides=(1,1),activation='relu'))
model.add(Conv2D(16,kernel_size=(1,10),strides=(1,1),activation='relu'))
model.add(Conv2D(16, kernel_size=(4,1),strides=(1,1),activation='relu'))
model.add(Conv2D(16, kernel_size=(4,1),strides=(1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(100,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3,activation='softmax'))
```

epochs = 100  
F1 test: 0.51              


* **2020-01-30** on colab  

FI_2010_cnn2_colab.ipynb

```
model = Sequential()
model.add(Conv2D(16, kernel_size=(1, 2),strides=(1,2),activation='relu')
model.add(Conv2D(16, kernel_size=(1,2),strides=(1,2),activation='relu'))
model.add(Conv2D(16,kernel_size=(1,10),strides=(1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(100,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(3,activation='softmax'))
```

epochs = 100  
F1 test: 0.51              


* **2020-01-30** on colab  

FI_2010_cnn.colab.ipynb

```
model = Sequential()
model.add(Conv2D(16, kernel_size=(4, 40),strides=(1,1),activation='relu')
model.add(Conv2D(16, kernel_size=(1,1),strides=(1,1),activation='relu'))
model.add(Conv2D(32,kernel_size=(4,1),strides=(1,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1),strides=2))
model.add(Conv2D(32,kernel_size=(3,1),strides=(1,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1),strides=2))
model.add(Flatten())
model.add(Dense(100,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(3,activation='softmax'))
```

epochs = 50  
F1 test: 0.53              
