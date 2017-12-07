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
  - Customize Small Conv Model (MobileNet + speech command example network)
    - Sufficiently high accuracy
    - Low computational complexity
    - Low energy usage
    - Small model size
    
  - cnn-trad-fpool3 (Paper : Convolutional Neural Networks for Small-footprint Keyword Spotting, Tara N. Sainath, Carolina Parada, 2017) customizing
    - Hyperparameter tuning
    - Change model architecture
  - Apply other CNN model
    - SqueezeNet (SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE, Forrest N. Iandola1, Song Han2, Matthew W. Moskewicz1, Khalid Ashraf1, William J. Dally2, Kurt Keutzer1, ICRL 2017)
    - MobileNet (MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam, 2017)
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


