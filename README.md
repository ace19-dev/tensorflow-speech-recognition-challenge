# Kaggle Competitions

## Team name
- Awesome-101
- A4
- AIR Jordan
- Walking through the forest of Deep Learning
- Fantastic 4

## TensorFlow Speech Recognition Challenge
- https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
- In this competition, you're challenged to use the Speech Commands Dataset to build an algorithm that understands simple spoken commands.


## Data
- The dataset is designed to let you build basic but useful voice interfaces for applications, with common words like “Yes”, “No”, digits, and directions included. 


### Data File Descriptions
- train.7z - Contains a few informational files and a folder of audio files. The audio folder contains subfolders with 1 second clips of voice commands, with the folder name being the label of the audio clip. There are more labels that should be predicted. The labels you will need to predict in Test are yes, no, up, down, left, right, on, off, stop, go. Everything else should be considered either unknown or silence. The folder _background_noise_ contains longer clips of "silence" that you can break up and use as training input.
  - The files contained in the training audio are not uniquely named across labels, but they are unique if you include the label folder. For example, 00f0204f_nohash_0.wav is found in 14 folders, but that file is a different speech command in each folder.
  - The files are named so the first element is the subject id of the person who gave the voice command, and the last element indicated repeated commands. Repeated commands are when the subject repeats the same word multiple times. Subject id is not provided for the test data, and you can assume that the majority of commands in the test data were from subjects not seen in train.
  - You can expect some inconsistencies in the properties of the training data (e.g., length of the audio).
- test.7z - Contains an audio folder with 150,000+ files in the format clip_000044442.wav. The task is to predict the correct label. Not all of the files are evaluated for the leaderboard score.
- sample_submission.csv - A sample submission file in the correct format.
- link_to_gcp_credits_form.txt - Provides the URL to request $500 in GCP credits, provided to the first 500 requestors.


### Evaluation
- Submissions are evaluated on Multiclass Accuracy, which is simply the average number of observations with the correct label.
- For audio clip in the test set, you must predict the correct label


### Prerequisite
- Google Research Blog Post announcing the Speech Commands Dataset. Note that much of what is provided as part of the training set is already public. However, the test set is not. (https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html)
- TensorFlow Audio Recognition Tutorial (https://www.tensorflow.org/versions/master/tutorials/audio_recognition)
- Link to purchase Raspberry Pi 3 on Amazon. This will be at your own expense.
- Also review the Prizes tab for details and tools for how the special prize will be evaluated. (https://www.kaggle.com/c/tensorflow-speech-recognition-challenge#Prizes)

## Question
- 어떤 방법으로 음성 데이터를 분류할 수 있을까??
- <b>대표적인 네트웍인 CNN 을 사용한다면, 어떻게 음성 데이터를 전처리할 수 있을까?? (main idea: Wave -> spectrogram)</b>
- 시간축이 포함된 데이터 인데 RNN 네트웍을 사용하는것이 더 좋은 방법이 아닐까??
- 또 다른 방법은 어떤것이 있을까? (WaveNet)
- 학습데이터를 손실없이 효과적으로 줄일 수 있는 방법은?
- 실 생활 데이터에서 음성만 추출할 수 있는 방법은?


## Solutions
- http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
- https://svds.com/tensorflow-rnn-tutorial/
	- github: https://github.com/silicon-valley-data-science/RNN-Tutorial
	- Several key improvements that have been made by the Microsoft team and other researchers in the past 4 years include:
		- using language models on top of character based RNNs
		- using convolutional neural nets (CNNs) for extracting features from the audio
		- ensemble models that utilize multiple RNNs
	- Computation and Language
		- Towards End-to-End Speech Recognition with Deep Convolutional Neural Network
			- https://arxiv.org/abs/1701.02720
		- The Microsoft 2016 Conversational Speech Recognition System
			- https://arxiv.org/abs/1609.03528
- https://deepmind.com/blog/wavenet-generative-model-raw-audio/
	- Pixel RNN
		- https://arxiv.org/abs/1601.06759
	- Pixel CNN
		- https://arxiv.org/abs/1606.05328
