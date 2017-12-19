## TensorFlow RNN Tutorial (https://svds.com/tensorflow-rnn-tutorial/)
- we’ll provide a short tutorial for training a RNN for speech recognition; we’re including code snippets throughout, and you can find the accompanying https://github.com/silicon-valley-data-science/RNN-Tutorial
- if you are brand new to RNNs, we highly recommend you read Christopher Olah’s excellent overview of RNN Long Short-Term Memory (LSTM) networks http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Speech recognition in the past and today both rely on decomposing sound waves into frequency and amplitude using fourier transforms, yielding a spectrogram
- https://github.com/silicon-valley-data-science/RNN-Tutorial
- Several key improvements that have been made by the Microsoft team and other researchers in the past 4 years include:
  - using language models on top of character based RNNs
  - using convolutional neural nets (CNNs) for extracting features from the audio
  - ensemble models that utilize multiple RNNs