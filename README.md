# Kaggle-Competitions-2nd
## TensorFlow Speech Recognition Challenge
- https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
- In this competition, you're challenged to use the Speech Commands Dataset to build an algorithm that understands simple spoken commands.


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
- Google Research Blog Post announcing the Speech Commands Dataset. Note that much of what is provided as part of the training set is already public. However, the test set is not.
- TensorFlow Audio Recognition Tutorial
- Link to purchase Raspberry Pi 3 on Amazon. This will be at your own expense.
- Also review the Prizes tab for details and tools for how the special prize will be evaluated.
