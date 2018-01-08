## How does this Model Work?
- The architecture used in this tutorial is based on some described in the paper Convolutional Neural Networks for Small-footprint Keyword Spotting. 
  - http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
  - It was chosen because it's comparatively simple, quick to train, and easy to understand.
- It's solved that issue by defining a window of time we believe our spoken words should fit into, and converting the audio signal in that window into an image.
  - This is done by grouping the incoming audio samples into short segments, just a few milliseconds long, and calculating the strength of the frequencies across a set of bands.
  - Each set of frequency strengths from a segment is treated as a vector of numbers, and those vectors are arranged in time order to form a two-dimensional array.
  - This array of values can then be treated like a single-channel image, and is known as a spectrogram.
- It's been traditional in speech recognition to do further processing to this representation to turn it into a set of Mel-Frequency Cepstral Coefficients, or MFCCs for short. 
  - This is also a two-dimensional, one-channel representation so it can be treated like an image too.
  - If you're targeting general sounds rather than speech you may find you can skip this step and operate directly on the spectrograms.


## To make more advanced speech systems
- We must limit the overall computation.
  - We limit the number of multiplications.
    - Limit the number of multiplies is to have one convolutional layer rather than two conv layer
    - Have the time filter span all of time.
  - We limit the number of parameters.
- Must have a small memory footprint and low computational power.

### preprocessing data
- you can also supply your own training data
- <b>If you have clips with variable amounts of silence at the start, you can look at word alignment tools to standardize them</b>
  - https://petewarden.com/2017/07/17/a-quick-hack-to-align-single-word-audio-recordings/
- <b>Image augmentation and audio augmentation.</b>
- Add Background Noise
  - add background noise (helps add some realism to the training)
  - you can supply your own audio clips in the _background_noise_ folder
- Use Silence/Unknown Class
- time shifting
  - Increasing this value will provide more variation, but at the risk of cutting off important parts of the audio.
  - Audio time stretching and pitch scaling
- various parameters to customize
  - tweak the spectrogram creation parameters
  - If you make the input smaller, the model will need fewer computations to process it
  - high enough sample rate
  - --window_size_ms
  - --window_stride_ms
  - --dct_coefficient_count
  - --clip_duration_ms
  
### Customizing the Model
- low_latency_conv: Based on the 'cnn-one-fstride4' topology described in the [Convolutional Neural Networks for Small-footprint Keyword Spotting paper](http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf)
- low_latency_svdf: Based on the topology presented in the [Compressing Deep Neural Networks using a Rank-Constrained Topology paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf)
- Apply other SOTA ConvNet.

### hyper-parameter tuning
- filter
  - number of filter
  - shape of filter
  - stride and zero-padding
- Learning Rate
- Cost Function
- Regularization parameter
- Multi batch size
- Traning repeat
- Hidden unit/layer
- Weight initialization