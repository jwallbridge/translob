# Experiment logbook for FI-2010

## In progress


## Completed

In reverse chronological order of completion time:

* **2019-08-10** on Church  
```
transformer2_100v4.ipynb
2-SIMPLICIAL MODEL
Batch size = 1024
Maximum length = 256
Model dimension = 64
2-simplicial dimension = 48
Number of heads = 4
Transformer depth = 6
Number of virtual entities = 2
```
```
2-simplicial encoder = True
Intermediate residual = True
qkk-normalization = None
Outer 2-simplex normalization = True * 1/sqrt(d2_model)  
```
```
2-simplicial decoder = False
```
```
2-simplicial encoder-decoder = False  
```
Case-insensitive results: ~10.5           
Case-sensitive results: ~10.5   


* **2020-01-30** on colab  
```
model = Sequential()
model.add(Conv2D(16, kernel_size=(4, 40),strides=(1,1),data_format='channels_last',activation='relu',
                 input_shape=(configs['sequence_length'],len(colx),1)))
model.add(Conv2D(16, kernel_size=(1,1),strides=(1,1),data_format='channels_last',activation='relu'))
model.add(Conv2D(32,kernel_size=(4,1),strides=(1,1),data_format='channels_last',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1),strides=2))
model.add(Conv2D(32,kernel_size=(3,1),strides=(1,1),data_format='channels_last',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1),strides=2))
model.add(Flatten())
model.add(Dense(100,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(3,activation='softmax'))
```
```
epochs = 50  
```

F1 test: 0.53              
