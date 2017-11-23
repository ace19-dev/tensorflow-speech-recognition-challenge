## Question
- 어떤 방법으로 음성 데이터를 분류할 수 있을까??
- 대표적인 네트웍인 CNN 을 사용한다면, 어떻게 음성 데이터를 전처리할 수 있을까?? (main point: Wave -> spectrogram)
- 비록 짧은 단어지만 시간 feature 가 포함되어 있는데, RNN 계열의 네트웍을 사용하는것이 더 좋은 방법이 아닐까??
- 또 다른 방법은 어떤것이 있을까?


## Reference
- http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
- https://svds.com/tensorflow-rnn-tutorial/
- https://deepmind.com/blog/wavenet-generative-model-raw-audio/


## How does this Model Work?
- This tutorial is based on the kind of convolutional network that will feel very familiar to anyone who's worked with image recognition.
- We solve that issue by defining a window of time we believe our spoken words should fit into, and converting the audio signal in that window into an image. 
  - this is done by grouping the incoming audio samples into short segments, just a few milliseconds long, calculating the strength of the frequencies across a set of bands
  - Each set of frequency strengths from a segment is treated as a vector of numbers, and those vectors are arranged in time order to form a two-dimensional array.
  - This array of values can then be treated like a single-channel image, and is known as a spectrogram
  - you can run the `wav_to_spectrogram tool:
    - bazel run tensorflow/examples/wav_to_spectrogram:wav_to_spectrogram -- \
       --input_wav=/tmp/speech_dataset/happy/ab00c4b2_nohash_0.wav \
       --output_png=/tmp/spectrogram.png
- Because the human ear is more sensitive to some frequencies than others, it's been traditional in speech recognition to do further processing to this representation to turn it into a set of Mel-Frequency Cepstral Coefficients, or MFCCs for short
- f you're targeting general sounds rather than speech you may find you can skip this step and operate directly on the spectrograms.
- <b>The image that's produced by these processing steps is then fed into a multi-layer convolutional neural network, with a fully-connected layer followed by a softmax at the end.</b>
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands


## # spectrogram
- A spectrogram is a visual representation of the spectrum of frequencies of sound or other signal as they vary with time or some other variable
- Spectrograms can be used to identify spoken words phonetically, and to analyse the various calls of animals. They are used extensively in the development of the fields of music, sonar, radar, and speech processing,[1] seismology, and others.

