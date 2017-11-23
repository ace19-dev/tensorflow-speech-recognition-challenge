## Question
- 어떤 방법으로 음성 데이터를 분류할 수 있을까??
- 대표적인 네트웍인 CNN 을 사용한다면, 어떻게 음성 데이터를 전처리할 수 있을까?? (main point: Wave -> spectrogram)
- 비록 짧은 단어지만 시간 feature 포함되어 있는데, RNN 네트웍을 사용하는것이 더 좋은 방법이 아닐까??
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
  - 


## spectrogram
- A spectrogram is a visual representation of the spectrum of frequencies of sound or other signal as they vary with time or some other variable
- Spectrograms can be used to identify spoken words phonetically, and to analyse the various calls of animals. They are used extensively in the development of the fields of music, sonar, radar, and speech processing,[1] seismology, and others.

