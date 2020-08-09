"""
Contains implementation of the Transformer model described in papers
"Attention is all you need" (https://arxiv.org/abs/1706.03762) and
"Universal Transformer" (https://arxiv.org/abs/1807.03819)
"""
# Based on Transformer from Keras-RL
# https://github.com/kpot/keras-transformer/blob/master/keras_transformer/transformer.py

import math
from typing import Union, Callable, Optional

from keras.layers import Layer, Add, Dropout
from tensorflow.keras.layers import InputSpec
from keras import initializers, constraints, activations
from keras import backend as K
from keras.utils import get_custom_objects

from tLobAttention import MultiHeadSelfAttention


class LayerNormalization(Layer):
    """
    Implementation of Layer Normalization (https://arxiv.org/abs/1607.06450).
    """
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(
            K.square(inputs - mean), axis=self.axis, keepdims=True)
        epsilon = K.constant(1e-5, dtype=K.floatx())
        normalized_inputs = (inputs - mean) / K.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class TransformerTransition(Layer):
    """
    Transformer transition function. The same function is used both
    in classical in Universal Transformers.
    """

    def __init__(self, activation: Union[str, Callable],
                 size_multiplier: int = 4, **kwargs):
        """
        :param activation: activation function. Must be a string or a callable.
        :param size_multiplier: How big the hidden dimension should be.
          Most of the implementation use transition functions having 4 times
          more hidden units than the model itself.
        :param kwargs: Keras-specific layer arguments.
        """
        self.activation = activations.get(activation)
        self.size_multiplier = size_multiplier
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['activation'] = activations.serialize(self.activation)
        config['size_multiplier'] = self.size_multiplier
        return config

    def build(self, input_shape):
        d_model = input_shape[-1]
        self.weights1 = self.add_weight(
            name='weights1',
            shape=(d_model, self.size_multiplier * d_model),
            initializer='glorot_uniform',
            trainable=True)
        self.biases1 = self.add_weight(
            name='biases1',
            shape=(self.size_multiplier * d_model,),
            initializer='zeros',
            trainable=True)
        self.weights2 = self.add_weight(
            name='weights2',
            shape=(self.size_multiplier * d_model, d_model),
            initializer='glorot_uniform',
            trainable=True)
        self.biases2 = self.add_weight(
            name='biases2',
            shape=(d_model,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        d_model = input_shape[-1]
        step1 = self.activation(
            K.bias_add(
                K.dot(K.reshape(inputs, (-1, d_model)),
                      self.weights1),
                self.biases1,
                data_format='channels_last'))
        step2 = K.bias_add(
            K.dot(step1, self.weights2),
            self.biases2,
            data_format='channels_last')
        result = K.reshape(step2, (-1,) + input_shape[-2:])
        return result


class TransformerBlock:
    """
    A pseudo-layer combining together all nuts and bolts to assemble
    a complete section of both the Transformer and the Universal Transformer
    models, following description from the "Universal Transformers" paper.
    Each such block is, essentially:
    - Multi-head self-attention (masked or unmasked)
    - Residual connection,
    - Layer normalization
    - Transition function
    - Residual connection
    - Layer normalization
    """
    def __init__(self, name: str, num_heads: int,
                 use_masking: bool = True):
        self.attention_layer = MultiHeadSelfAttention(
            num_heads, use_masking=use_masking, 
            name=f'{name}_self_attention')
        self.norm1_layer = LayerNormalization(name=f'{name}_normalization1')
        self.norm2_layer = LayerNormalization(name=f'{name}_normalization2')
        self.transition_layer = TransformerTransition(
            name=f'{name}_transition', activation='relu')
        self.addition_layer = Add(name=f'{name}_add')

    def __call__(self, _input):
        output = self.attention_layer(_input)
        post_residual1 = (
            self.addition_layer([_input, output]))
        norm1_output = self.norm1_layer(post_residual1)
        output = self.transition_layer(norm1_output)
        post_residual2 = (
            self.addition_layer([norm1_output, output]))
        output = self.norm2_layer(post_residual2)
        return output
        
        
        
        
        
        
