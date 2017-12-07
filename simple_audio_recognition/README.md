## Simple Audio Recognition tutorial
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands
- It's solved that issue by defining a window of time we believe our spoken words should fit into, and converting the audio signal in that window into an image.
- This is done by grouping the incoming audio samples into short segments, just a few milliseconds long, and calculating the strength of the frequencies across a set of bands.
- Each set of frequency strengths from a segment is treated as a vector of numbers, and those vectors are arranged in time order to form a two-dimensional array.
- This array of values can then be treated like a single-channel image, and is known as a spectrogram.
- do further processing to this representation to turn it into a set of Mel-Frequency Cepstral Coefficients, or MFCCs for short.


## How does this Model Work?
- The architecture used in this tutorial is based on some described in the paper Convolutional Neural Networks for Small-footprint Keyword Spotting. (http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf)
- we must limit the overall computation
  - we limit the number of multiplications 
    - limit the number of multiplies is to have one convolutional layer rather than two conv layer
    - have the time filter span all of time.
  - we limit the number of parameters
- Must have a small memory footprint and low computational power.
- spectral representations of speech have strong correlations in time and frequency


## tuning point
- To make more advanced speech systems..

### customizing the model - by Kim-SungIl, hongym7 (~목요일)
- create custom conv (low_latency_conv, low_latency_svdf)
- Other parameters to customize
  - tweak the spectrogram creation parameters
  - If you make the input smaller, the model will need fewer computations to process it
  - --window_stride_ms
  - --dct_coefficient_count
  - --window_size_ms
  - --clip_duration_ms
- Implement other conv (by using recent paper)

### custom training data - by ziippy (~목요일)
- you can also supply your own training data
- <b>you can look at word alignment tools to standardize them (https://petewarden.com/2017/07/17/a-quick-hack-to-align-single-word-audio-recordings/)</b>
- image augmentation? audio augmentation?
- Unknown Class
- Background Noise
  - add background noise (helps add some realism to the training)
  - you can supply your own audio clips in the _background_noise_ folder
- Silence
- time shifting
  - Audio time stretching and pitch scaling

### hyper-parameter tuning - by hongym7
- Learning Rate
- Cost Function
- Regularization parameter
- Multi batch size
- Traning repeat
- Hidden unit/layer
- Weight initialization



