# Kaggle Competitions

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
  - There are only 12 possible labels for the Test set: yes, no, up, down, left, right, on, off, stop, go, silence, unknown.
  - The unknown label should be used for a command that is not one one of the first 10 labels or that is not silence.
- For audio clip in the test set, you must predict the correct label
- The submission file should contain a header and have the following format:


### Prerequisite
- Google Research Blog Post announcing the Speech Commands Dataset. Note that much of what is provided as part of the training set is already public. However, the test set is not. (https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html)
- TensorFlow Audio Recognition Tutorial (https://www.tensorflow.org/versions/master/tutorials/audio_recognition)
- Link to purchase Raspberry Pi 3 on Amazon. This will be at your own expense.
- Also review the Prizes tab for details and tools for how the special prize will be evaluated. (https://www.kaggle.com/c/tensorflow-speech-recognition-challenge#Prizes)


## Question
- How can we classify voice data?
- <b> If you use CNN, a typical network, how can you preprocess voice data? (main idea: Wave -> spectrogram) </b>
- Is it better to use the RNN network because it contains data with time axis?
- What else is there? (WaveNet)
- How can we effectively reduce learning data without loss?
- How can I extract voice only from real life data?


## Solutions 
### First we will use this -> (https://www.tensorflow.org/versions/master/tutorials/audio_recognition)
- Customize Small ConvNet Models
  - cnn-trad-fpool3 (paper : http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf)
    - Hyperparameter tuning
    - Change model architecture (add layer, change filter size, etc..)
  - Apply other SOTA CNN models
    - MobileNet (paper : https://arxiv.org/pdf/1704.04861.pdf)
      - 3x3 depthwise separable convolutions - between 8 to 9 times less computations.
      - Width Multiplier & Resolution Multiplier - less computations.
    - SqueezeNet (paper : https://arxiv.org/pdf/1602.07360.pdf)
      - Replace 3x3 filters with 1x1 filters
      - Decrease the number of input channels to 3x3 filters
      - Downsample late in the network so that convolution layers have large activation maps
    - etc..
- Data pre-processing (Prepare a best spectrogram image for learning)
  - wav volume normalization
  - find the section of the word based on volume dB level efficiently
  - create the spectrogram png using by wav_to_spectrogram
  - each spectrogram png size change to same size
  - data augmentation

## Result
- We eventually reached 250th with a score of 0.86019. - Public Leaderboard
- We recommed to see the 9th (0.90637) participant's blog. 
  - http://openresearch.ai/t/ideas-for-9th-kaggle-tensorflow-speech-recognition-challenge/105 (korean version)
    - i am sure you get wonderful ideas.
  - https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47618
    - 1D conv.

## Quick Start
- It's simple. 
  - set your own parameter
  - python new_train.py

## Additional work 
### After reaching a satisfactory level, we might try other resolutions
- see [reference](https://github.com/ace19-dev/tensorflow-speech-recognition-challenge/tree/master/reference)


