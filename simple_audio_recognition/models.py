# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

import mobilenet

slim = tf.contrib.slim

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'single_fc':
    return create_single_fc_model(fingerprint_input, model_settings,
                                  is_training)
  elif model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'mobile':
    return create_low_layer_mobilenet_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'mobile2':
    return create_mobilenet_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'low_latency_conv':
    return create_low_latency_conv_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'low_latency_svdf':
    return create_low_latency_svdf_model(fingerprint_input, model_settings,
                                         is_training, runtime_settings)
  elif model_architecture == 'squeeze':
    return create_low_latency_squeeze_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'squeeze2':
    return create_low_latency_squeeze_model2(fingerprint_input, model_settings, is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, or "low_latency_svdf"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_single_fc_model(fingerprint_input, model_settings, is_training):
  """Builds a model with a single hidden fully-connected layer.

  This is a very simple model with just one matmul and bias layer. As you'd
  expect, it doesn't produce very accurate results, but it is very fast and
  simple, so it's useful for sanity testing.

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  weights = tf.Variable(
      tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
  bias = tf.Variable(tf.zeros([label_count]))
  logits = tf.matmul(fingerprint_input, weights) + bias
  if is_training:
    return logits, dropout_prob
  else:
    return logits


def create_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a mobilenet model.
  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])

  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  #first_conv = BatchNorm(first_conv, is_training, name='bn1')
  first_relu = tf.nn.relu(first_conv)
  #first_relu = LeakyReLU(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias

  #second_conv = BatchNorm(second_conv, is_training, name='bn2')
  second_relu = tf.nn.relu(second_conv)
  #second_relu = LeakyReLU(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2] # 20
  second_conv_output_height = second_conv_shape[1] # 33

  # second_conv_element_count = 42240
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)

  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])

  # label_count = 12 = x + 2
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_layer_mobilenet_model(fingerprint_input, model_settings, is_training):
  """
        Conv / s2

        BN

        Relu

        Conv dw / s1

        BN

        Relu

        Conv / s1

        BN

        Relu

        Avg Pooling


  """

  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])

  print('... ')
  print('... ')

  modify_input = tf.image.resize_bilinear(fingerprint_4d, [224, 224])

  print('after modify_input', modify_input)


  # Conv / s2

  first_filter_width = 3
  first_filter_height = 3
  first_filter_count = 32

  first_weights = tf.get_variable("first_weights",
    shape=[first_filter_height, first_filter_width, 1, first_filter_count],
    initializer=tf.contrib.layers.xavier_initializer())

  print('after first_weights', first_weights)

  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(modify_input, first_weights, [1, 2, 2, 1], 'SAME') + first_bias

  #first_bn = slim.batch_norm(first_conv)
  first_bn = BatchNorm(first_conv, is_training, name='bn1')
  first_relu = tf.nn.relu(first_bn)

  print('after first_relu', first_relu)


  # Conv dw / s1

  deepwise_filter_width = 3
  deepwise_filter_height = 3
  deepwise_filter_channel = 32
  deepwise_filter_count = 32

  second_weights = tf.get_variable("second_weights",
    shape=[deepwise_filter_height, deepwise_filter_width, deepwise_filter_channel, deepwise_filter_count],
    initializer=tf.contrib.layers.xavier_initializer())

  second_bias = tf.Variable(tf.zeros([deepwise_filter_count]))

  second_conv = tf.nn.conv2d(first_relu, second_weights, [1, 1, 1, 1],
                            'SAME') + second_bias

  #second_bn = slim.batch_norm(second_conv)
  second_bn = BatchNorm(second_conv, is_training, name='bn2')
  second_relu = tf.nn.relu(second_bn)

  print('after second_relu', second_relu)

  one_filter_width = 1
  one_filter_height = 1
  third_filter_count = 64

  third_weights = tf.get_variable("third_weights",
    shape=[one_filter_height, one_filter_width, deepwise_filter_count, third_filter_count],
    initializer=tf.contrib.layers.xavier_initializer())

  third_bias = tf.Variable(tf.zeros([third_filter_count]))

  third_conv = tf.nn.conv2d(second_relu, third_weights, [1, 1, 1, 1],
                             'SAME') + third_bias

  #third_bn = slim.batch_norm(third_conv)
  third_bn = BatchNorm(third_conv, is_training, name='bn3')
  third_relu = tf.nn.relu(third_bn)

  print('after third_relu', third_relu)

  fourth_weights = tf.get_variable("fourth_weights",
    shape=[deepwise_filter_height, deepwise_filter_width, 64, 64],
    initializer=tf.contrib.layers.xavier_initializer())

  fourth_bias = tf.Variable(tf.zeros([64]))

  fourth_conv = tf.nn.conv2d(third_relu, fourth_weights, [1, 2, 2, 1],
                             'SAME') + fourth_bias

  #fourth_bn = slim.batch_norm(fourth_conv)
  fourth_bn = BatchNorm(fourth_conv, is_training, name='bn4')
  fourth_relu = tf.nn.relu(fourth_bn)

  print('after fourth_relu', fourth_relu)

  fifth_weights = tf.get_variable("fifth_weights",
    shape=[one_filter_height, one_filter_width, 64, 128],
    initializer=tf.contrib.layers.xavier_initializer())

  fifth_bias = tf.Variable(tf.zeros([128]))

  fifth_conv = tf.nn.conv2d(fourth_relu, fifth_weights, [1, 1, 1, 1],
                             'SAME') + fifth_bias


  #fifth_bn = slim.batch_norm(fifth_conv)
  fifth_bn = BatchNorm(fifth_conv, is_training, name='bn5')
  fifth_relu = tf.nn.relu(fifth_bn)

  print('after fifth_relu', fifth_relu)

  sixth_weights = tf.get_variable("sixth_weights",
    shape=[deepwise_filter_height, deepwise_filter_width, 128, 128],
    initializer=tf.contrib.layers.xavier_initializer())

  sixth_bias = tf.Variable(tf.zeros([128]))

  sixth_conv = tf.nn.conv2d(fifth_relu, sixth_weights, [1, 1, 1, 1],
                             'SAME') + sixth_bias

  #sixth_bn = slim.batch_norm(sixth_conv)
  sixth_bn = BatchNorm(sixth_conv, is_training, name='bn6')
  sixth_relu = tf.nn.relu(sixth_bn)

  print('after sixth_relu', sixth_relu)

  seventh_weights = tf.get_variable("seventh_weights",
    shape=[one_filter_height, one_filter_width, 128, 128],
    initializer=tf.contrib.layers.xavier_initializer())

  seventh_bias = tf.Variable(tf.zeros([128]))

  seventh_conv = tf.nn.conv2d(sixth_relu, seventh_weights, [1, 1, 1, 1],
                             'SAME') + seventh_bias


  #seventh_bn = slim.batch_norm(seventh_conv)
  seventh_bn = BatchNorm(seventh_conv, is_training, name='bn7')
  seventh_relu = tf.nn.relu(seventh_bn)

  print('after seventh_relu', seventh_relu)


  eighth_weights = tf.get_variable("eighth_weights",
    shape=[deepwise_filter_height, deepwise_filter_width, 128, 128],
    initializer=tf.contrib.layers.xavier_initializer())

  eighth_bias = tf.Variable(tf.zeros([128]))

  eighth_conv = tf.nn.conv2d(seventh_relu, eighth_weights, [1, 2, 2, 1],
                             'SAME') + sixth_bias

  #eighth_bn = slim.batch_norm(eighth_conv)
  eighth_bn = BatchNorm(eighth_conv, is_training, name='bn8')
  eighth_relu = tf.nn.relu(eighth_bn)

  print('after eighth_relu', eighth_relu)





  
  nineth_weights = tf.get_variable("nineth_weights",
    shape=[one_filter_height, one_filter_width, 128, 256],
    initializer=tf.contrib.layers.xavier_initializer())

  nineth_bias = tf.Variable(tf.zeros([256]))

  nineth_conv = tf.nn.conv2d(eighth_relu, nineth_weights, [1, 1, 1, 1],
                             'SAME') + nineth_bias


  #nineth_bn = slim.batch_norm(nineth_conv)
  nineth_bn = BatchNorm(nineth_conv, is_training, name='bn9')
  nineth_relu = tf.nn.relu(nineth_bn)

  print('after nineth_relu', nineth_relu)







  tenth_weights = tf.get_variable("tenth_weights",
    shape=[deepwise_filter_height, deepwise_filter_width, 256, 256],
    initializer=tf.contrib.layers.xavier_initializer())

  tenth_bias = tf.Variable(tf.zeros([256]))

  tenth_conv = tf.nn.conv2d(nineth_relu, tenth_weights, [1, 1, 1, 1],
                             'SAME') + tenth_bias

  #tenth_bn = slim.batch_norm(tenth_conv)
  tenth_bn = BatchNorm(tenth_conv, is_training, name='bn10')
  tenth_relu = tf.nn.relu(tenth_bn)

  print('after tenth_relu', tenth_relu)





  eleventh_weights = tf.get_variable("eleventh_weights",
    shape=[one_filter_height, one_filter_width, 256, 256],
    initializer=tf.contrib.layers.xavier_initializer())

  eleventh_bias = tf.Variable(tf.zeros([256]))

  eleventh_conv = tf.nn.conv2d(tenth_relu, eleventh_weights, [1, 1, 1, 1],
                             'SAME') + eleventh_bias


  #eleventh_bn = slim.batch_norm(eleventh_conv)
  eleventh_bn = BatchNorm(eleventh_conv, is_training, name='bn11')
  eleventh_relu = tf.nn.relu(eleventh_bn)

  print('after eleventh_relu', eleventh_relu)





  twelfth_weights = tf.get_variable("twelfth_weights",
    shape=[deepwise_filter_height, deepwise_filter_width, 256, 256],
    initializer=tf.contrib.layers.xavier_initializer())

  twelfth_bias = tf.Variable(tf.zeros([256]))

  twelfth_conv = tf.nn.conv2d(eleventh_relu, twelfth_weights, [1, 2, 2, 1],
                             'SAME') + twelfth_bias

  #twelfth_bn = slim.batch_norm(twelfth_conv)
  twelfth_bn = BatchNorm(twelfth_conv, is_training, name='bn12')
  twelfth_relu = tf.nn.relu(twelfth_bn)

  print('after twelfth_relu', twelfth_relu)






  thirteenth_weights = tf.get_variable("thirteenth_weights",
    shape=[one_filter_height, one_filter_width, 256, 512],
    initializer=tf.contrib.layers.xavier_initializer())

  thirteenth_bias = tf.Variable(tf.zeros([512]))

  thirteenth_conv = tf.nn.conv2d(twelfth_relu, thirteenth_weights, [1, 1, 1, 1],
                             'SAME') + thirteenth_bias


  #thirteenth_bn = slim.batch_norm(thirteenth_conv)
  thirteenth_bn = BatchNorm(thirteenth_conv, is_training, name='bn13')
  thirteenth_relu = tf.nn.relu(thirteenth_bn)

  print('after thirteenth_relu', thirteenth_relu)





  fourteenth_weights = tf.get_variable("fourteenth_weights",
    shape=[deepwise_filter_height, deepwise_filter_width, 512, 512],
    initializer=tf.contrib.layers.xavier_initializer())

  fourteenth_bias = tf.Variable(tf.zeros([512]))

  fourteenth_conv = tf.nn.conv2d(thirteenth_relu, fourteenth_weights, [1, 1, 1, 1],
                             'SAME') + fourteenth_bias

  #fourteenth_bn = slim.batch_norm(fourteenth_conv)
  fourteenth_bn = BatchNorm(fourteenth_conv, is_training, name='bn14')
  fourteenth_relu = tf.nn.relu(fourteenth_bn)

  print('after fourteenth_relu', fourteenth_relu)



  fifteenth_weights = tf.get_variable("fifteenth_weights",
    shape=[one_filter_height, one_filter_width, 512, 512],
    initializer=tf.contrib.layers.xavier_initializer())

  fifteenth_bias = tf.Variable(tf.zeros([512]))

  fifteenth_conv = tf.nn.conv2d(fourteenth_relu, fifteenth_weights, [1, 1, 1, 1],
                             'SAME') + fifteenth_bias


  #fifteenth_bn = slim.batch_norm(fifteenth_conv)
  fifteenth_bn = BatchNorm(fifteenth_conv, is_training, name='bn15')
  fifteenth_relu = tf.nn.relu(fifteenth_bn)

  print('after fifteenth_relu', fifteenth_relu)





  sixteenth_weights = tf.get_variable("sixteenth_weights",
    shape=[deepwise_filter_height, deepwise_filter_width, 512, 512],
    initializer=tf.contrib.layers.xavier_initializer())

  sixteenth_bias = tf.Variable(tf.zeros([512]))

  sixteenth_conv = tf.nn.conv2d(fifteenth_relu, sixteenth_weights, [1, 2, 2, 1],
                             'SAME') + sixteenth_bias

  #sixteenth_bn = slim.batch_norm(sixteenth_conv)
  sixteenth_bn = BatchNorm(sixteenth_conv, is_training, name='bn16')
  sixteenth_relu = tf.nn.relu(sixteenth_bn)

  print('after sixteenth_relu', sixteenth_relu)






  seventeenth_weights = tf.get_variable("seventeenth_weights",
    shape=[one_filter_height, one_filter_width, 512, 1024],
    initializer=tf.contrib.layers.xavier_initializer())

  seventeenth_bias = tf.Variable(tf.zeros([1024]))

  seventeenth_conv = tf.nn.conv2d(sixteenth_relu, seventeenth_weights, [1, 1, 1, 1],
                             'SAME') + seventeenth_bias


  #seventeenth_bn = slim.batch_norm(seventeenth_conv)
  seventeenth_bn = BatchNorm(seventeenth_conv, is_training, name='bn17')
  seventeenth_relu = tf.nn.relu(seventeenth_bn)

  print('after seventeenth_relu', seventeenth_relu)



  eighteenth_weights = tf.get_variable("eighteenth_weights",
    shape=[deepwise_filter_height, deepwise_filter_width, 1024, 1024],
    initializer=tf.contrib.layers.xavier_initializer())

  eighteenth_bias = tf.Variable(tf.zeros([1024]))

  eighteenth_conv = tf.nn.conv2d(seventeenth_relu, eighteenth_weights, [1, 1, 1, 1],
                             'SAME') + eighteenth_bias

  #eighteenth_bn = slim.batch_norm(eighteenth_conv)
  eighteenth_bn = BatchNorm(eighteenth_conv, is_training, name='bn18')
  eighteenth_relu = tf.nn.relu(eighteenth_bn)

  print('after eighteenth_relu', eighteenth_relu)



  nineteenth_weights = tf.get_variable("nineteenth_weights",
    shape=[one_filter_height, one_filter_width, 1024, 1024],
    initializer=tf.contrib.layers.xavier_initializer())

  nineteenth_bias = tf.Variable(tf.zeros([1024]))

  nineteenth_conv = tf.nn.conv2d(eighteenth_relu, nineteenth_weights, [1, 1, 1, 1],
                             'SAME') + nineteenth_bias


  #nineteenth_bn = slim.batch_norm(nineteenth_conv)
  nineteenth_bn = BatchNorm(nineteenth_conv, is_training, name='bn19')
  nineteenth_relu = tf.nn.relu(nineteenth_bn)

  print('after nineteenth_relu', nineteenth_relu)








  avg_pool = tf.nn.avg_pool(nineteenth_relu, [1, 7, 7, 1], [1, 1, 1, 1], 'VALID')

  last_conv_shape = avg_pool.get_shape()
  last_conv_output_width = last_conv_shape[2]
  last_conv_output_height = last_conv_shape[1]



  print('after avg_pool', avg_pool)

  # second_conv_element_count = 42240
  # change last arg
  last_conv_element_count = int(
      last_conv_output_width * last_conv_output_height *
      1024)


  flattened_last_conv = tf.reshape(avg_pool,
                                     [-1, last_conv_element_count])

  # flattened_last_conv = (?, 10368)
  print('after flattened_last_conv', flattened_last_conv)

  # label_count = 12 = x + 2
  label_count = model_settings['label_count']


  print('last_conv_element_count : ', last_conv_element_count)


  final_fc_weights = tf.get_variable("final_fc_weights",
    shape=[last_conv_element_count, label_count],
    initializer=tf.contrib.layers.xavier_initializer())

  final_fc_bias = tf.Variable(tf.zeros([label_count]))

  final_fc = tf.matmul(flattened_last_conv, final_fc_weights) + final_fc_bias

  print('final_fc : ', final_fc)

  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_mobilenet_model(fingerprint_input, model_settings, is_training):
  dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size,
                                                  input_frequency_size, 1])
  modify_fingerprint_4d = tf.image.resize_bilinear(fingerprint_4d, [224, 224])

  logits, end_points = mobilenet.mobilenet(modify_fingerprint_4d, model_settings['label_count'])
  return logits, dropout_prob


def create_low_latency_conv_model(fingerprint_input, model_settings,
                                  is_training):
  """Builds a convolutional model with low compute requirements.

  This is roughly the network labeled as 'cnn-one-fstride4' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This produces slightly lower quality results than the 'conv' model, but needs
  fewer weight parameters and computations.

  During training, dropout nodes are introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = input_time_size
  first_filter_count = 186
  first_filter_stride_x = 1
  first_filter_stride_y = 1
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
      1, first_filter_stride_y, first_filter_stride_x, 1
  ], 'VALID') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  first_conv_output_width = math.floor(
      (input_frequency_size - first_filter_width + first_filter_stride_x) /
      first_filter_stride_x)
  first_conv_output_height = math.floor(
      (input_time_size - first_filter_height + first_filter_stride_y) /
      first_filter_stride_y)
  first_conv_element_count = int(
      first_conv_output_width * first_conv_output_height * first_filter_count)
  flattened_first_conv = tf.reshape(first_dropout,
                                    [-1, first_conv_element_count])
  first_fc_output_channels = 128
  first_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_conv_element_count, first_fc_output_channels], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 128
  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_fc_output_channels, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_svdf_model(fingerprint_input, model_settings,
                                  is_training, runtime_settings):
  """Builds an SVDF model with low compute requirements.

  This is based in the topology presented in the 'Compressing Deep Neural
  Networks using a Rank-Constrained Topology' paper:
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

  Here's the layout of the graph:

  (fingerprint_input)
          v
        [SVDF]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  This model produces lower recognition accuracy than the 'conv' model above,
  but requires fewer weight parameters and, significantly fewer computations.

  During training, dropout nodes are introduced after the relu, controlled by a
  placeholder.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    The node is expected to produce a 2D Tensor of shape:
      [batch, model_settings['dct_coefficient_count'] *
              model_settings['spectrogram_length']]
    with the features corresponding to the same time slot arranged contiguously,
    and the oldest slot at index [:, 0], and newest at [:, -1].
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
      ValueError: If the inputs tensor is incorrectly shaped.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']

  # Validation.
  input_shape = fingerprint_input.get_shape()
  if len(input_shape) != 2:
    raise ValueError('Inputs to `SVDF` should have rank == 2.')
  if input_shape[-1].value is None:
    raise ValueError('The last dimension of the inputs to `SVDF` '
                     'should be defined. Found `None`.')
  if input_shape[-1].value % input_frequency_size != 0:
    raise ValueError('Inputs feature dimension %d must be a multiple of '
                     'frame size %d', fingerprint_input.shape[-1].value,
                     input_frequency_size)

  # Set number of units (i.e. nodes) and rank.
  rank = 2
  num_units = 1280
  # Number of filters: pairs of feature and time filters.
  num_filters = rank * num_units
  # Create the runtime memory: [num_filters, batch, input_time_size]
  batch = 1
  memory = tf.Variable(tf.zeros([num_filters, batch, input_time_size]),
                       trainable=False, name='runtime-memory')
  # Determine the number of new frames in the input, such that we only operate
  # on those. For training we do not use the memory, and thus use all frames
  # provided in the input.
  # new_fingerprint_input: [batch, num_new_frames*input_frequency_size]
  if is_training:
    num_new_frames = input_time_size
  else:
    window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                           model_settings['sample_rate'])
    num_new_frames = tf.cond(
        tf.equal(tf.count_nonzero(memory), 0),
        lambda: input_time_size,
        lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))
  new_fingerprint_input = fingerprint_input[
      :, -num_new_frames*input_frequency_size:]
  # Expand to add input channels dimension.
  new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

  # Create the frequency filters.
  weights_frequency = tf.Variable(
      tf.truncated_normal([input_frequency_size, num_filters], stddev=0.01))
  # Expand to add input channels dimensions.
  # weights_frequency: [input_frequency_size, 1, num_filters]
  weights_frequency = tf.expand_dims(weights_frequency, 1)
  # Convolve the 1D feature filters sliding over the time dimension.
  # activations_time: [batch, num_new_frames, num_filters]
  activations_time = tf.nn.conv1d(
      new_fingerprint_input, weights_frequency, input_frequency_size, 'VALID')
  # Rearrange such that we can perform the batched matmul.
  # activations_time: [num_filters, batch, num_new_frames]
  activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

  # Runtime memory optimization.
  if not is_training:
    # We need to drop the activations corresponding to the oldest frames, and
    # then add those corresponding to the new frames.
    new_memory = memory[:, :, num_new_frames:]
    new_memory = tf.concat([new_memory, activations_time], 2)
    tf.assign(memory, new_memory)
    activations_time = new_memory

  # Create the time filters.
  weights_time = tf.Variable(
      tf.truncated_normal([num_filters, input_time_size], stddev=0.01))
  # Apply the time filter on the outputs of the feature filters.
  # weights_time: [num_filters, input_time_size, 1]
  # outputs: [num_filters, batch, 1]
  weights_time = tf.expand_dims(weights_time, 2)
  outputs = tf.matmul(activations_time, weights_time)
  # Split num_units and rank into separate dimensions (the remaining
  # dimension is the input_shape[0] -i.e. batch size). This also squeezes
  # the last dimension, since it's not used.
  # [num_filters, batch, 1] => [num_units, rank, batch]
  outputs = tf.reshape(outputs, [num_units, rank, -1])
  # Sum the rank outputs per unit => [num_units, batch].
  units_output = tf.reduce_sum(outputs, axis=1)
  # Transpose to shape [batch, num_units]
  units_output = tf.transpose(units_output)

  # Appy bias.
  bias = tf.Variable(tf.zeros([num_units]))
  first_bias = tf.nn.bias_add(units_output, bias)

  # Relu.
  first_relu = tf.nn.relu(first_bias)

  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu

  first_fc_output_channels = 256
  first_fc_weights = tf.Variable(
      tf.truncated_normal([num_units, first_fc_output_channels], stddev=0.01))
  first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
  first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 256
  second_fc_weights = tf.Variable(
      tf.truncated_normal(
          [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
  second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_fc_output_channels, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_squeeze_model(fingerprint_input, model_settings, is_training):
    squeeze_ratio = 1
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])
    print('fingerprint_4d : ', fingerprint_4d)

    #fingerprint_3d = tf.reshape(fingerprint_input, [input_time_size, input_frequency_size, -1])
    #normal_input = tf.image.per_image_standardization(fingerprint_3d)
    #fingerprint_4d = tf.reshape(normal_input, [-1, input_time_size, input_frequency_size, 1])


    modify_fingerprint_4d = tf.image.resize_bilinear(fingerprint_4d,[224,224])
    print('fingerprint_4d : ', modify_fingerprint_4d)
    first_filter_width = 7
    first_filter_height = 7
    first_filter_count = 64
    first_weights = tf.get_variable("first_weight", shape=[first_filter_height, first_filter_width, 1, first_filter_count],
                                    initializer=tf.contrib.layers.xavier_initializer())

    # conv1_1
    first_conv = tf.nn.conv2d(modify_fingerprint_4d, first_weights, [1, 2, 2, 1], 'SAME')
    print('first_conv : ', first_conv)
    relu1 = tf.nn.relu(first_conv + bias_variable([64]))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print('pool1 : ', pool1)
    fire2 = fire_module('fire2', pool1, squeeze_ratio * 16, 64, 64)
    print('fire2 : ', fire2)
    fire3 = fire_module('fire3', fire2, squeeze_ratio * 16, 64, 64, True)
    print('fire3 : ', fire3)
    pool3 = tf.nn.max_pool(fire3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print('pool3 : ', pool3)
    fire4 = fire_module('fire4', pool3, squeeze_ratio * 32, 128, 128)
    print('fire4 : ', fire4)
    fire5 = fire_module('fire5', fire4, squeeze_ratio * 32, 128, 128, True)
    print('fire5 : ', fire5)
    pool5 = tf.nn.max_pool(fire5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    fire6 = fire_module('fire6', pool5, squeeze_ratio * 48, 192, 192)

    fire7 = fire_module('fire7', fire6, squeeze_ratio * 48, 192, 192, True)

    fire8 = fire_module('fire8', fire7, squeeze_ratio * 64, 256, 256)

    fire9 = fire_module('fire9', fire8, squeeze_ratio * 64, 256, 256, True)

    # 50% dropout
    dropout9 = tf.nn.dropout(fire9, dropout_prob)
    print('dropout9 : ', dropout9)
    second_weights = tf.Variable(tf.random_normal([1, 1, 512, 1000], stddev=0.01), name="second_weight")
    second_conv = tf.nn.conv2d(dropout9, second_weights, [1, 1, 1, 1], 'SAME')
    print('second_conv : ', second_conv)
    relu10 = tf.nn.relu(second_conv + bias_variable([1000]))
    print('relu10 : ', relu10)
    # avg pool
    pool10 = tf.nn.avg_pool(relu10, ksize=[1, 13, 13, 1], strides=[1, 1, 1, 1], padding='VALID')
    print('pool10 : ', pool10)
    last_conv_shape = pool10.get_shape()
    last_conv_ouput_width = last_conv_shape[2]
    last_conv_ouput_height = last_conv_shape[1]
    last_conv_element_count = int(last_conv_ouput_width * last_conv_ouput_height * 1000)
    flattend_last_conv = tf.reshape(pool10, [-1, last_conv_element_count])
    label_count = model_settings['label_count']
    print('last_conv_element_count', last_conv_element_count)
    print('flattend_last_conv', flattend_last_conv)
    print('label_count', label_count)
    final_fc_weights = tf.get_variable("final_fc_weights", shape=[last_conv_element_count, label_count], initializer=tf.contrib.layers.xavier_initializer())
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattend_last_conv, final_fc_weights) + final_fc_bias

    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc

def create_low_latency_squeeze_model2(fingerprint_input, model_settings, is_training):
    squeeze_ratio = 1
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size, 1])
    print('fingerprint_4d : ', fingerprint_4d)

    modify_fingerprint_4d = tf.image.resize_bilinear(fingerprint_4d,[224,224])
    print('fingerprint_4d : ', modify_fingerprint_4d)
    first_filter_width = 7
    first_filter_height = 7
    first_filter_count = 64
    first_weights = tf.get_variable("first_weight", shape=[first_filter_height, first_filter_width, 1, first_filter_count],
                                    initializer=tf.contrib.layers.xavier_initializer())

    # conv1_1
    first_conv = tf.nn.conv2d(modify_fingerprint_4d, first_weights, [1, 2, 2, 1], 'SAME')
    print('first_conv : ', first_conv)
    relu1 = tf.nn.relu(first_conv + bias_variable([64]))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print('pool1 : ', pool1)
    fire2 = fire_module('fire2', pool1, squeeze_ratio * 16, 64, 64)
    print('fire2 : ', fire2)
    fire3 = fire_module('fire3', fire2, squeeze_ratio * 16, 64, 64, True)
    print('fire3 : ', fire3)
    fire4 = fire_module('fire4', fire3, squeeze_ratio * 32, 128, 128)
    print('fire4 : ', fire4)

    pool4 = tf.nn.max_pool(fire4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print('pool4 : ', pool4)

    fire5 = fire_module('fire5', pool4, squeeze_ratio * 32, 128, 128, True)
    print('fire5 : ', fire5)
    fire6 = fire_module('fire6', fire5, squeeze_ratio * 48, 192, 192)

    fire7 = fire_module('fire7', fire6, squeeze_ratio * 48, 192, 192, True)

    fire8 = fire_module('fire8', fire7, squeeze_ratio * 64, 256, 256)

    pool5 = tf.nn.max_pool(fire8, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    fire9 = fire_module('fire9', pool5, squeeze_ratio * 64, 256, 256, True)

    # 50% dropout
    dropout9 = tf.nn.dropout(fire9, dropout_prob)

    print('dropout9 : ', dropout9)
    second_weights = tf.Variable(tf.random_normal([1, 1, 512, 1000], stddev=0.01), name="second_weight")
    second_conv = tf.nn.conv2d(dropout9, second_weights, [1, 1, 1, 1], 'SAME')
    print('second_conv : ', second_conv)
    relu10 = tf.nn.relu(second_conv + bias_variable([1000]))
    print('relu10 : ', relu10)
    # avg pool
    pool10 = tf.nn.avg_pool(relu10, ksize=[1, 13, 13, 1], strides=[1, 1, 1, 1], padding='VALID')
    print('pool10 : ', pool10)
    last_conv_shape = pool10.get_shape()
    last_conv_ouput_width = last_conv_shape[2]
    last_conv_ouput_height = last_conv_shape[1]
    last_conv_element_count = int(last_conv_ouput_width * last_conv_ouput_height * 1000)
    flattend_last_conv = tf.reshape(pool10, [-1, last_conv_element_count])
    label_count = model_settings['label_count']
    print('last_conv_element_count', last_conv_element_count)
    print('flattend_last_conv', flattend_last_conv)
    print('label_count', label_count)
    final_fc_weights = tf.get_variable("final_fc_weights", shape=[last_conv_element_count, label_count], initializer=tf.contrib.layers.xavier_initializer())
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattend_last_conv, final_fc_weights) + final_fc_bias

    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def bias_variable(shape, value=0.1):
    return tf.Variable(tf.constant(value, shape=shape))


def fire_module(layer_name, layer_input, s1x1, e1x1, e3x3, residual=False):
    """ Fire module consists of squeeze and expand convolutional layers. """
    fire = {}
    shape = layer_input.get_shape()
    # squeeze
    s1_weight = tf.get_variable(layer_name + '_s1', shape=[1, 1, int(shape[3]), s1x1], initializer=tf.contrib.layers.xavier_initializer())
    # expand
    e1_weight = tf.get_variable(layer_name + '_e1', shape=[1, 1, s1x1, e1x1], initializer=tf.contrib.layers.xavier_initializer())
    e3_weight = tf.get_variable(layer_name + '_e3', shape=[3, 3, s1x1, e3x3], initializer=tf.contrib.layers.xavier_initializer())

    fire['s1'] = tf.nn.conv2d(layer_input, s1_weight, strides=[1, 1, 1, 1], padding='SAME')
    fire['relu1'] = tf.nn.relu(fire['s1'] + bias_variable([s1x1]))

    fire['e1'] = tf.nn.conv2d(fire['relu1'], e1_weight, strides=[1, 1, 1, 1], padding='SAME')
    fire['e3'] = tf.nn.conv2d(fire['relu1'], e3_weight, strides=[1, 1, 1, 1], padding='SAME')

    fire['concat'] = tf.concat([tf.add(fire['e1'], bias_variable([e1x1])),
                                tf.add(fire['e3'], bias_variable([e3x3]))], 3)
    if residual:
        fire['relu2'] = tf.nn.relu(tf.add(fire['concat'], layer_input))
    else:
        fire['relu2'] = tf.nn.relu(fire['concat'])
    return fire['relu2']

## Regularizations
def BatchNorm(input, is_train, decay=0.999, name='BatchNorm'):
  '''
  https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/layers.py
  https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
  http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow
  '''
  from tensorflow.python.training import moving_averages
  from tensorflow.python.ops import control_flow_ops

  axis = list(range(len(input.get_shape()) - 1))
  fdim = input.get_shape()[-1:]

  with tf.variable_scope(name):
    beta = tf.get_variable('beta', fdim, initializer=tf.constant_initializer(value=0.0))
    gamma = tf.get_variable('gamma', fdim, initializer=tf.constant_initializer(value=1.0))
    moving_mean = tf.get_variable('moving_mean', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)
    moving_variance = tf.get_variable('moving_variance', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)

    def mean_var_with_update():
      batch_mean, batch_variance = tf.nn.moments(input, axis)
      update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=True)
      update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay, zero_debias=True)
      with tf.control_dependencies([update_moving_mean, update_moving_variance]):
        return tf.identity(batch_mean), tf.identity(batch_variance)
    
    #mean, variance = control_flow_ops.cond(tf.cast(is_train, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))

    if is_train:
      mean, variance = mean_var_with_update()
    else:
      mean, variance = moving_mean, moving_variance

  return tf.nn.batch_normalization(input, mean, variance, beta, gamma, 1e-3)


def LeakyReLU(input, alpha=0.2):
  return tf.maximum(input, alpha*input)
