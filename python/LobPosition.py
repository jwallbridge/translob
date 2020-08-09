import numpy as np
from keras import backend as K
from keras.engine import Layer
from keras.utils import get_custom_objects
from keras.layers import Concatenate


def positional_encoding(x):
  steps, d_model = x.get_shape().as_list()[-2:] 
  ps = np.zeros([steps,1],dtype=K.floatx()) 
  for tx in range(steps):
    ps[tx,:] = [(2/(steps-1))*tx - 1] 

  ps_expand = K.expand_dims(K.constant(ps),axis=0) 
  ps_tiled = K.tile(ps_expand,[K.shape(x)[0],1,1]) 

  x = K.concatenate([x,ps_tiled],axis=-1)
  return x