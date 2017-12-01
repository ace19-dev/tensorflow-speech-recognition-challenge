## Simple Audio Recognition tutorial
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands

## How does this Model Work?
- The architecture used in this tutorial is based on some described in the paper Convolutional Neural Networks for Small-footprint Keyword Spotting. (http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf)
  - we limit the number of multiplications
  - we limit the number of parameters

## tuning point
- For more advanced speech systems, I recommend looking into Kaldi(?)

### hyperparameter tuning
- learning rate
- batch size

### custom training data 
- create custom training data
  - augmentation
- add background noise
- time shifting
- Other parameters to customize
  - You'll need to make sure that all your training data contains the right audio in the initial portion of the clip though.

### customizing the model
- create custom conv
- low_latency_conv
- low_latency_svdf
