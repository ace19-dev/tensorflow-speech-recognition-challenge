## Simple Audio Recognition tutorial
- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands

## tuning point
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
