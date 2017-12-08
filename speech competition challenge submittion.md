## 팀명
- AIR-Jordan

## 대표자 이름/소속
- 김성연/SK 주식회사 C&C

## 대표자 연락처 (필수: 핸드폰번호, 이메일)
- 010-8836-7348, 
- acemc19@gmail.com

## 참여 팀원 소개 (필수: 이름, 약력, 연락처, 개인별 github 주소)
| 이름 | 약력 | 연락처 | github 주소 |
|---|---|---|---|
| 김성연 | 2011~현재: SK 주식회사 C&C 융합서비스개발그룹 수석 | 010-8836-7348  | https://github.com/ace19-dev  |
| 김종성 | 2016~현재: SK 주식회사 C&C 융합서비스개발그룹 수석 | 010-2601-3608  | https://github.com/ziippy  |
| 홍용만 | 2011~현재: SK 주식회사 C&C 융합서비스개발그룹 선임 | 010-2882-7354  | https://github.com/hongym7  |
| 김성일 | 2016~현재: SK 주식회사 C&C 융합서비스개발그룹 선임 | 010-3275-2578  | https://github.com/Kim-SungIl  |


## 제안하는 알고리즘 (어떤 방향으로 문제를 풀 예정인가요?)
- We are looking for two workarounds.
  - Customize Small Conv Model
    - cnn-trad-fpool3 (paper : http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf)
      - Hyperparameter tuning
      - Change model architecture (add layer, change filter size...)
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


