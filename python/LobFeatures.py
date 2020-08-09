from keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Reshape
from keras.layers import Dropout, Activation, Lambda
from keras.layers import LSTM, Conv1D, Conv2D, Flatten
from keras.layers import MaxPooling1D, MaxPooling2D, Reshape, BatchNormalization


def lob_cnn(x):
  """
  DeepLOB cnn module
  """
  y = layers.Conv2D(16, kernel_size=(1, 2),strides=(1,2),data_format='channels_last')(x)
  y = layers.LeakyReLU(alpha=0.01)(y)
  y = layers.Conv2D(16, kernel_size=(4,1),strides=(1,1),data_format='channels_last')(y)
  y = layers.LeakyReLU(alpha=0.01)(y)
  y = layers.Conv2D(16, kernel_size=(4,1),strides=(1,1),data_format='channels_last')(y)
  y = layers.LeakyReLU(alpha=0.01)(y)

  y = layers.Conv2D(16, kernel_size=(1,2),strides=(1,2),data_format='channels_last')(y)
  y = layers.LeakyReLU(alpha=0.01)(y)
  y = layers.Conv2D(16, kernel_size=(4,1),strides=(1,1),data_format='channels_last')(y)
  y = layers.LeakyReLU(alpha=0.01)(y)
  y = layers.Conv2D(16, kernel_size=(4,1),strides=(1,1),data_format='channels_last')(y)
  y = layers.LeakyReLU(alpha=0.01)(y)

  y = layers.Conv2D(16, kernel_size=(1,10),strides=(1,2),data_format='channels_last')(y)
  y = layers.LeakyReLU(alpha=0.01)(y)
  y = layers.Conv2D(16, kernel_size=(4,1),strides=(1,1),data_format='channels_last')(y)
  y = layers.LeakyReLU(alpha=0.01)(y)
  y = layers.Conv2D(16, kernel_size=(4,1),strides=(1,1),data_format='channels_last')(y)
  y = layers.LeakyReLU(alpha=0.01)(y)

 # y = layers.Dropout(rate=0.2)(y)
  
  return y

def lob_inception(x):
  """
  DeepLOB inception module
  """
  branch_a = layers.Conv2D(32, kernel_size=(1,1))(x)
  branch_a = layers.LeakyReLU(alpha=0.01)(branch_a)
  branch_a = layers.Conv2D(32, kernel_size=(3,1))(branch_a)
  branch_a = layers.LeakyReLU(alpha=0.01)(branch_a)

  branch_b = layers.Conv2D(32, kernel_size=(1,1))(x)
  branch_b = layers.LeakyReLU(alpha=0.01)(branch_b)
  branch_b = layers.Conv2D(32, kernel_size=(5,1))(branch_b)
  branch_b = layers.LeakyReLU(alpha=0.01)(branch_b)

  branch_c = layers.MaxPooling2D(pool_size=(3,1))(x)
  branch_c = layers.Conv2D(32, kernel_size=(1,1))(branch_c)
  branch_c = layers.LeakyReLU(alpha=0.01)(branch_c)

  y = layers.concatenate([branch_a, branch_b, branch_c], axis=1)
  
  return y

def lob_dilated(x):
  """
  TransLOB dilated 1-D convolution module
  """
  x = layers.Conv1D(14,kernel_size=2,strides=1,activation='relu',padding='causal')(x)   
  x = layers.Conv1D(14,kernel_size=2,dilation_rate=2,activation='relu',padding='causal')(x)
  x = layers.Conv1D(14,kernel_size=2,dilation_rate=4,activation='relu',padding='causal')(x)
  x = layers.Conv1D(14,kernel_size=2,dilation_rate=8,activation='relu',padding='causal')(x)
  y = layers.Conv1D(14,kernel_size=2,dilation_rate=16,activation='relu',padding='causal')(x)

  return y
