# Experiment logbook for FI-2010

## Future 

* **2020-02-00** on gcp 

FI_2010_translob_gcp.ipynb  

```
TransLOB (250)
```

k=10  
epochs = 125


* **2020-02-00** on gcp 

FI_2010_translob.ipynb  

```
TransLOB (32)
```

k=10  
epochs = 125

* **2020-02-00** on gcp  

FI_2010_deeplob_gcp.ipynb  

```
DeepLOB (32)
```

k=10  
epochs = 125



## Completed

In reverse chronological order of completion time:

* **2020-02-07** on colab  

FI_2010_translob_colab_v3.ipynb     
d_model = 16  
num_heads = 3  
num_blocks = 2  

```
TransLOB (32)
```
        
epochs = 150            
Training time : ~6hrs         
F1 test: 0.91        


* **2020-02-07** on colab  

FI_2010_translob_colab_v2.ipynb (epochs = 100)    
FI_2010_translob_redux_v2.ipynb (epochs = 50)  
d_model = 16  
num_heads = 3  
num_blocks = 2  

```
TransLOB (250)
```
        
epochs = 150            
Training time : ~3hrs         
F1 test: 0.89        


* **2020-02-06** on colab  

FI_2010_translob_colab_v1.ipynb  
d_model = 30  
num_heads = 1  
num_blocks = 1  

```
TransLOB (250)
```
        
epochs = 100          
Training time : ~3hrs       
F1 test: 0.86      


* **2020-02-03** on colab  

FI_2010_deeplob_redux.ipynb  
saved_model : 03022020-015756-e40.h5  
Graph : events.out.tfevents.1580695078.f91ade03496a

```
DeepLOB (250)
```

Continuation of DeepLOB after the previous disconnected at 62 epochs        
epochs = 40 (so total ~100 epochs)        
Training time : ~8hrs    
Total training time : ~20hrs    
F1 test: 0.75      
  

* **2020-01-31** on colab  

FI_2010_deeplob_colab.ipynb  

```
DeepLOB (250)
```

k=10, batch_size = 250        
Training time : colab disconnected after ~12 hours      
Results at epoch = 62 :  loss: 0.7536 - acc: 0.6581 - val_loss: 0.7710 - val_acc: 0.6601  
   

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

k=10  
epochs = 100  
Total training time : ~3hrs  
F1 test: 0.64    

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

k=10  
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

k=10  
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

k=10  
epochs = 50  
F1 test: 0.53              
