import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine import Layer
from keras.utils import get_custom_objects
# Based on MultiHeadSelfAttention from Keras-RL
# https://github.com/kpot/keras-transformer/blob/master/keras_transformer/attention.py

class MultiHeadSelfAttention(Layer):
    """
    Base class for Multi-head Self-Attention layers.
    """
    def __init__(self, num_heads: int, use_masking: bool,
                 **kwargs):
        """
        :param num_heads: number of attention heads
        :param use_masking: when True, forbids the attention to see the further
          elements in the sequence.
        :param kwargs: any extra arguments typical for a Keras layer,
          such as name, etc.
        """
        self.num_heads = num_heads
        self.use_masking = use_masking
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['num_heads'] = self.num_heads
        config['use_masking'] = self.use_masking
        return config

    def build(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError('Invalid input')
        d_model = input_shape[-1]
        
        self.validate_model_dimensionality(d_model)
        self.qkv_weights = self.add_weight(
            name='qkv_weights',
            shape=(d_model, d_model * 3),  # * 3 for q, k and v
            initializer='glorot_uniform',
            trainable=True)
        
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not K.is_tensor(inputs):
            raise ValueError(
                'The layer can be called only with one tensor as an argument')
        _, seq_len, d_model = K.int_shape(inputs)
        
        # Perform affine transformations to get the Queries, the Keys and the Values.
        qkv = K.dot(inputs, self.qkv_weights) # (-1,seq_len,d_model*3)
        qkv = K.reshape(qkv,[-1,d_model*3])

        # splitting the keys, the values and the queries.
        pre_q, pre_k, pre_v = [
            K.reshape(
                qkv[:, i * d_model:(i + 1) * d_model],
                (-1, seq_len, self.num_heads, d_model // self.num_heads))
            for i in range(3)]
        
        attention_out = self.attention(pre_q, pre_v, pre_k, seq_len, d_model,
                                       training=kwargs.get('training'))
        # of shape (-1, seq_len, d_model)
        return attention_out

    def compute_output_shape(self, input_shape):
        shape_a, seq_len, d_model = input_shape
        return (shape_a, seq_len, d_model)
    
    def validate_model_dimensionality(self, d_model: int):
        if d_model % self.num_heads != 0:
            raise ValueError(
                f'The size of the last dimension of the input '
                f'({d_model}) must be evenly divisible by the number'
                f'of the attention heads {self.num_heads}')
    
    def attention(self, pre_q, pre_v, pre_k, seq_len: int, d_model: int,
                  training=None):
        """
        Calculates the output of the attention once the affine transformations
        of the inputs are done. Here's the shapes of the arguments:
        :param pre_q: (batch_size, q_seq_len, num_heads, d_model // num_heads)
        :param pre_v: (batch_size, v_seq_len, num_heads, d_model // num_heads)
        :param pre_k: (batch_size, k_seq_len, num_heads, d_model // num_heads)
        :param seq_len: the length of the output sequence
        :param d_model: dimensionality of the model (by the paper)
        :param training: Passed by Keras. Should not be defined manually.
          Optional scalar tensor indicating if we're in training
          or inference phase.
        """
        d_submodel = d_model // self.num_heads
        
        # shaping Q and V into (batch_size, num_heads, seq_len, d_model//heads)
        q = K.permute_dimensions(pre_q, [0, 2, 1, 3])
        v = K.permute_dimensions(pre_v, [0, 2, 1, 3])
        k = K.permute_dimensions(pre_k, [0, 2, 3, 1])
        
        q = K.reshape(q, (-1,seq_len,d_submodel))
        k = K.reshape(k, (-1,seq_len,d_submodel))
        v = K.reshape(v, (-1,seq_len,d_submodel))
        
        qk = tf.einsum('aib,ajb->aij', q, k)
        sqrt_d = K.constant(np.sqrt(d_model // self.num_heads),
                            dtype=K.floatx())
        a = qk/sqrt_d
        a = self.mask_attention(a)
        a = K.softmax(a)
        attention_heads = tf.einsum('aij,ajb->aib', a, v)
        
        attention_heads = K.reshape(attention_heads, (-1, self.num_heads, seq_len, d_model))
        attention_heads = K.permute_dimensions(attention_heads, [0, 2, 1, 3])
        attention_heads = K.reshape(attention_heads,(-1, seq_len, d_model))

        return attention_heads

    def mask_attention(self, dot_product):
        """
        Makes sure that (when enabled) each position
        (of a decoder's self-attention) cannot attend to subsequent positions.
        :param dot_product: scaled dot-product of Q and K after reshaping them
        to 3D tensors (batch * num_heads, rows, cols)
        """
        if not self.use_masking:
            return dot_product
        last_dims = K.int_shape(dot_product)[-2:]
        low_triangle_ones = (
            np.tril(np.ones(last_dims))
            # to ensure proper broadcasting
            .reshape((1,) + last_dims))
        inverse_low_triangle = 1 - low_triangle_ones
        close_to_negative_inf = -1e9
        result = (
            K.constant(low_triangle_ones, dtype=K.floatx()) * dot_product +
            K.constant(close_to_negative_inf * inverse_low_triangle))
        return result


get_custom_objects().update({
    'MultiHeadSelfAttention': MultiHeadSelfAttention,
})
