
- We are looking for two workarounds.
  - Customize Small Conv Model
    - cnn-trad-fpool3 (paper : http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf)
      - Hyperparameter tuning
      - Change model architecture (add layer, change filter size etc.)
    - Apply other SOTA CNN models
      - MobileNet (paper : https://arxiv.org/pdf/1704.04861.pdf)
        - 3x3 depthwise separable convoutions - between 8 to 9 times less computations.
        - Width Multiplier & Resolution Multiplier - less computations.
      - SqueezeNet (paper : https://arxiv.org/pdf/1602.07360.pdf)
        - Replace 3x3 filters with 1x1 filters
        - Decrease the number of input channels to 3x3 filters
        - Downsample late in the network so that convolution layers have large activation maps
      - etc..
  
  - data pre-processing (Prepare a best spectrogram image for learing)
    - wav volume normalization
    - find the section of the word based on volume dB level efficiently
    - create the spectorgram png using by wav_to_spectrogram
    - each spectrogram png size change to same size
    - Augmentation
      - pitch shift
      - time expanding
      - time_shift
      - How loud the background noise
      - Number of frequency bins to use for analysis


