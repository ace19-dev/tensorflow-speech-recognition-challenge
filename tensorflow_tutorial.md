## Question
- 어떤 방법으로 음성 데이터를 분류할 수 있을까??
- 분류하면 딥러닝에서 떠오르는 대표적인 네트웍인 CNN 을 사용할 수 있다면, 어떻게 음성 데이터를 전처리할 수 있을까??
- 비록 짧은 단어지만 시간 feature 포함되어 있는데, RNN 네트웍을 사용하는것이 더 좋은 방법이 아닐까??
- 또 다른 방법은 어떤것이 있을까?


## Overview
- This tutorial is based on the kind of convolutional network that will feel very familiar to anyone who's worked with image recognition.
- We solve that issue by defining a window of time we believe our spoken words should fit into, and converting the audio signal in that window into an image. 


## How does this Model Work?
- http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
- https://svds.com/tensorflow-rnn-tutorial/
- https://deepmind.com/blog/wavenet-generative-model-raw-audio/


