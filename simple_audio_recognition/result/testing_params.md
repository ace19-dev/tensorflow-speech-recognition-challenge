## tried
|  No | Testor  | Network | Valid Acc.|Test Acc. | Pred Acc.| epoch      | Learning Rate   | Batch | Optimizer  | Activation |win_stride|win_size  | time_shift|sample_rate  |back_vol.|back_freq.|dct_coef.|train_data|test_data|
|-----|---------|---------|---------- |----------|----------|------------|-----------------|-------|------------|------------|----------|----------|-----------|-------------|---------|----------|---------|---------|---------|
|  1  | Sean    |  conv   |  92.3%    |   91.6%  |   82%    | 9000,3000  | 0.001,0.0001    | 100   | Momentum   | Relu       | 10       |   30     | 100       |  16000      |  0.3    |  0.8     |   40    |   speech_dataset      |    audio    |
|  2  | Sean    |  conv   |  92.0%    |   92.6%  |   82%    | 9000,3000  | 0.002,0.0003    | 100   | Momentum   | Relu       | 10       |   30     | 100       |  16000      |  0.3    |  0.8     |   40    |   speech_dataset      |    audio    |
|  3  | joongjum| squeeze |  90.1%    |  90.4%   |   80%    | 6000,3000  | 0.001,0.0001    | 100   | RMSProp    | Relu       | 10       |   30     | 100       |  16000      |  0.3    |  0.9     |   40    |   speech_dataset      |    audio    |
|  4  | joongjum| squeeze |  91.3%    |  91.0%   |   80 %    | 7000,5000  | 0.001,0.0001    | 100   | RMSProp    | Relu       | 10       |   30     | 100       |  16000      |  0.1    |  0.8     |   40    |   speech_dataset      |    audio    |
|  5  | Sean    | squeeze |  91.3%    |  92.4%   |   79%    | 9000,3000  | 0.001,0.0001    | 100   | Adam       | Relu       | 10       |   30     | 100       |  16000      |  0.1    |  0.7     |   40    |   speech_dataset      |    audio    |
|  6  | Sean    | squeeze |  91.6%    |  91.1%   |   80%    | 9000,4000  | 0.001,0.0001    | 100   | Adam       | Relu       | 10       |   30     | 100       |  16000      |  0.3    |  0.9     |   40    |   speech_dataset      |    audio    |
|  7  | ziippy  | conv    |  86.8%  |  89.1%  |     72%     |   9000,3000  | 0.001,0.0001    | 100   | RMSProp   |  Relu      | 10       |   30     |  0        |  16000      |  0.1    |  0.8     |   40      |   speech_dataset_1200      |    audio_1200    |
|  8  | ziippy  | conv    |  85.5%  |  84.7%  |     00%     |  9000,3000  | 0.001,0.0001     | 100   | RMSProp   |  Relu      | 10       |   30     |  0        |  16000      |  0.1    |  0.8     |   40      |   speech_dataset_shift     |    audio     |
|  9  |         |         |           |          |          |            |                 |       |            |            |          |          |           |  16000      |         |          |         |         |        |


## tips
- validation accuracy/loss 그래프 변화를 살펴보고 오버피팅 여부를 파악하여 epoch 값을 정한다.
- tensorboard 를 이용한다.
