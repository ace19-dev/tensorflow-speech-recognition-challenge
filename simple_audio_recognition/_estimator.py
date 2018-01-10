# Now we want to predict testset and make submission file.
# Create datagenerator and input_function Load model Iterate over predictions and store results

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import csv

from tqdm import tqdm

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

import models
import new_input_data
import prediction_input_data

def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      len(new_input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)

  POSSIBLE_LABELS = new_input_data.prepare_words_list(FLAGS.wanted_words.split(','))
  params = dict(
    # seed=2018,
    # batch_size=FLAGS.batch_size,
    # keep_prob=0.5,
    # learning_rate=0.0002,
    # clip_gradients=15.0,
    # use_batch_norm=True,
    # num_classes=len(POSSIBLE_LABELS)
  )

  hparams = tf.contrib.training.HParams(**params)
  run_config = tf.contrib.learn.RunConfig()
  run_config = run_config.replace(model_dir=FLAGS.train_dir)

  audio_processor2 = prediction_input_data.AudioProcessor(
    FLAGS.data_dir,
    FLAGS.prediction_data_dir,
    model_settings
  )

  set_size = audio_processor2.set_size()
  print('prediction data size: ', set_size)
  def prediction_data_generator():
    def generator():
      for i in xrange(0, set_size, FLAGS.prediction_batch_size):
        # Pull the audio samples we'll use for testing.
        fname, fingerprints = \
          audio_processor2.get_data(FLAGS.prediction_batch_size, i,
                                    model_settings, 0.0, 0.0, 0, sess)

        yield dict(
          fname=fname,
          fingerprint_input=fingerprints
        )

    return generator

  test_input_fn = generator_input_fn(
    x=prediction_data_generator(),
    batch_size=FLAGS.prediction_batch_size,
    queue_capacity=10 * FLAGS.prediction_batch_size,
  )

  def model_fn(features, labels, mode, params):
    """Model function for Estimator."""
    logits = models.create_model(
      tf.cast(features['fingerprint_input'], tf.float32),
      model_settings,
      FLAGS.model_architecture,
      is_training=False)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
        'fname': tf.cast(features['fname'], tf.float32),
        'label': tf.argmax(logits, axis=-1)
      }
      specs = dict(
        mode=mode,
        predictions=predictions
      )
    return tf.estimator.EstimatorSpec(**specs)


  def get_estimator(config, hparams):
    """Return the model as a Tensorflow Estimator object.
    Args:
       run_config (RunConfig): Configuration for Estimator run.
       params (HParams): hyperparameters.
    """
    return tf.estimator.Estimator(
      model_fn=model_fn,
      config=config
      # params=hparams,
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

  # make submission.csv
  fout = open(os.path.join(FLAGS.result_dir, 'submission.csv'), 'w', encoding='utf-8', newline='')
  writer = csv.writer(fout)
  writer.writerow(['fname', 'label'])
  for key in sorted(submission.keys()):
    writer.writerow([key, submission[key]])
  fout.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='../../../dl_data/speech_commands/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--prediction_data_dir',
      type=str,
      default='../../../dl_data/speech_commands/test/audio/',
      help="""\
        Where is speech testing data.
        """)
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
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=15.0,
      help='How far to move in time between frequency windows',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
    '--prediction_batch_size',
    type=int,
    default=5000,
    help='How many items to train with at once', )
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
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')


  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
