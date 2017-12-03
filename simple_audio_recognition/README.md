## Simple Audio Recognition tutorial
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands
- It's solved that issue by defining a window of time we believe our spoken words should fit into, and converting the audio signal in that window into an image.
- This is done by grouping the incoming audio samples into short segments, just a few milliseconds long, and calculating the strength of the frequencies across a set of bands.
- Each set of frequency strengths from a segment is treated as a vector of numbers, and those vectors are arranged in time order to form a two-dimensional array.
- This array of values can then be treated like a single-channel image, and is known as a spectrogram.

## How does this Model Work?
- The architecture used in this tutorial is based on some described in the paper Convolutional Neural Networks for Small-footprint Keyword Spotting. (http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf)
- we must limit the overall computation
  - we limit the number of multiplications -> a novel CNN architecture which does not pool but rather strides the filter in frequency
    -  limit the number of multiplies is to have one convolutional layer rather than two
  - we limit the number of parameters
    - On way to improve CNN performance is to increase feature maps.
    -
- Must have a small memory footprint and low computational power.
- spectral representations of speech have strong correlations in time and frequency

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
