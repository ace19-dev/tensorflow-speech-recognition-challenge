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
r"""Simple speech recognition to spot a limited number of keywords.
This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.
It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".
To run the training process, use:
bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!
As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.
If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:
my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav
You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.
To pull this all together, you'd run:
bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import csv

from tqdm import tqdm

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
import test_data

from tensorflow.python.platform import gfile
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

FLAGS = None



def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings)
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  #fingerprint_size : 2600
  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  #shape of fingerprint_input : [2, ]
  logits, dropout_prob = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=True)

  # Define loss and optimizer
  ground_truth_input = tf.placeholder(
      tf.float32, [None, label_count], name='groundtruth_input')

  # shape of ground_truth_input : [2, ]
  # print(tf.shape(ground_truth_input))

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=logits))
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    momentum = tf.placeholder(tf.float32, [], name='momentum')
    # SGD
    train_step = \
      tf.train.MomentumOptimizer(learning_rate_input, momentum, use_nesterov=True).\
        minimize(cross_entropy_mean)

  predicted_indices = tf.argmax(logits, 1)
  expected_indices = tf.argmax(ground_truth_input, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)

  global_step = tf.contrib.framework.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  tf.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  tf.logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                       FLAGS.model_architecture + '.pbtxt')

  # Save list of words.
  with gfile.GFile(
      os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),'w') as f:
    f.write('\n'.join(audio_processor.words_list))

  # Training loop.
  training_steps_max = np.sum(training_steps_list)
  for training_step in xrange(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break
    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, 'training', sess)
    # Run the graph with this batch of training data.
    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [
            merged_summaries, evaluation_step, cross_entropy_mean, train_step,
            increment_global_step
        ],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            momentum:0.95,
            dropout_prob: 0.5
        })
    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                    (training_step, learning_rate_value, train_accuracy * 100,
                     cross_entropy_value))

    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                     0.0, 0, 'validation', sess))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy, conf_matrix = sess.run(
            [merged_summaries, evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
                dropout_prob: 1.0
            })
        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix
      tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))

    # Save the model checkpoint periodically.
    if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     FLAGS.model_architecture + '.ckpt')
      tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)


  # for testing
  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_prob: 1.0
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100, set_size))



  # for prediction
  tf.logging.info('Lets prediction >>> \n')
  POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
  params = dict(
    seed=2018,
    batch_size=FLAGS.prediction_batch_size,
    keep_prob=0.5,
    learning_rate=0.0002,
    clip_gradients=15.0,
    use_batch_norm=True,
    num_classes=len(POSSIBLE_LABELS)
  )

  hparams = tf.contrib.training.HParams(**params)
  # model_dir = './models'  # folder for model, checkpoints, logs and submission.csv
  run_config = tf.contrib.learn.RunConfig()
  run_config = run_config.replace(model_dir=FLAGS.train_dir)

  audio_processor2 = test_data.AudioProcessor(
    FLAGS.data_dir,
    FLAGS.test_data_dir,
    # FLAGS.silence_percentage,
    # FLAGS.unknown_percentage,
    FLAGS.wanted_words.split(','),
    model_settings
    # FLAGS.validation_percentage,
    # FLAGS.testing_percentage,
  )
  print('testing data size: ', audio_processor2.set_size('testing'))
  set_size = audio_processor2.set_size('testing')

  def test_data_generator():
    def generator():
      for i in xrange(0, set_size, FLAGS.prediction_batch_size):
        # Pull the audio samples we'll use for testing.
        fname, fingerprints = audio_processor2.get_data(
          FLAGS.prediction_batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)

        yield dict(
          fname=np.string_(fname),
          input_data=fingerprints
        )

    return generator

  test_input_fn = generator_input_fn(
    x=test_data_generator(),
    batch_size=hparams.batch_size,
    shuffle=False,
    num_epochs=1,
    queue_capacity=10 * hparams.batch_size,
    num_threads=1
  )

  def model_fn(features, labels, mode, params):
    """Model function for Estimator."""
    logits = models.create_model(
      tf.cast(features['input_data'], tf.float32),
      model_settings,
      FLAGS.model_architecture,
      is_training=False)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
        'fname': features['fname'],
        'label': tf.argmax(logits, axis=-1)
      }
      specs = dict(
        mode=mode,
        predictions=predictions
      )
    return tf.estimator.EstimatorSpec(**specs)

  def get_estimator(config=None, hparams=None):
    """Return the model as a Tensorflow Estimator object.
    Args:
       run_config (RunConfig): Configuration for Estimator run.
       params (HParams): hyperparameters.
    """
    return tf.estimator.Estimator(
      model_fn=model_fn,
      config=config,
      params=hparams,
    )

  estimator = get_estimator(config=run_config, hparams=hparams)
  it = estimator.predict(input_fn=test_input_fn)

  id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
  # last batch will contain padding, so remove duplicates
  submission = dict()
  for t in tqdm(it):
    fname, label = t['fname'].decode(), id2name[t['label']]
    # print("fname >>> : ", fname, ", ", "label >>> : ", label)
    submission[fname] = label

  # memory dump
  fout = open(os.path.join(FLAGS.result_dir, 'memory_dump.csv'), 'w', encoding='utf-8', newline='')
  writer = csv.writer(fout)
  for key in sorted(submission.keys()):
    # print("%s: %s" % (key, submission[key]))
    writer.writerow([key, submission[key]])
  fout.close()

  fin = open(os.path.join(FLAGS.result_dir, 'sample_submission.csv'), 'r', encoding='utf-8')
  reader = csv.reader(fin)
  fout = open(os.path.join(FLAGS.result_dir, 'submission.csv'), 'w', encoding='utf-8', newline='')
  writer = csv.writer(fout)
  for row in reader:
    if row[0] != 'fname':
      row[1] = submission[row[0]]
    writer.writerow(row)
  fin.close()
  fout.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='../../../dl_data/speech_commands/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
    '--test_data_dir',
    type=str,
    default='../../../dl_data/speech_commands/test/audio/',
    help="""\
          Where is speech testing data.
          """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.3,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.9,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=15,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=20.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How far to move in time between frequency windows',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='100',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=1000,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.002',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='./models/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='./models',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
    '--result_dir',
    type=str,
    default='./result',
    help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=500,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='mobile',
      help='What model architecture to use')
  parser.add_argument(
    '--prediction_batch_size',
    type=int,
    default=1,
    help='How many items to predict with at once', )
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
