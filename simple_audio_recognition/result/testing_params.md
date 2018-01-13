## tried
|  No | Testor  | Network | Valid Acc.|Test Acc. | Pred Acc.| epoch      | Learning Rate   | Batch | Optimizer  | win_stride|win_size  | time_shift|sample_rate  |back_vol.|back_freq.|dct_coef.|train_data                |test_data|
|-----|---------|---------|---------- |----------|----------|--------------|---------------|-------|----------|----------|----------|-----------|-------------|---------|----------|---------|---------------------------|------------|
|  1  | Sean    |  conv   |  92.3%    |   91.6%  |   82%    | 9000,3000  | 0.001,0.0001    | 100   | Momentum   |  10       |   30     | 100       |  16000      |  0.3    |  0.8     |   40    |   speech_dataset         |    audio    |
|  2  | Sean    |  conv   |  92.0%    |   92.6%  |   82%    | 9000,3000  | 0.002,0.0003    | 100   | Momentum   |  10       |   30     | 100       |  16000      |  0.3    |  0.8     |   40    |   speech_dataset         |    audio    |
|  3  | joongjum| squeeze |  90.1%    |  90.4%   |   80%    | 6000,3000  | 0.001,0.0001    | 100   | RMSProp    |  10       |   30     | 100       |  16000      |  0.3    |  0.9     |   40    |   speech_dataset         |    audio    |
|  4  | joongjum| squeeze |  91.3%    |  91.0%   |   80 %    | 7000,5000  | 0.001,0.0001   | 100   | RMSProp    |  10       |   30     | 100       |  16000      |  0.1    |  0.8    |   40    |   speech_dataset        |    audio    |
|  5  | Sean    | squeeze |  91.3%    |  92.4%   |   79%    | 9000,3000  | 0.001,0.0001    | 100   | Adam       |  10       |   30     | 100       |  16000      |  0.1    |  0.7     |   40    |   speech_dataset         |    audio    |
|  6  | Sean    | squeeze |  91.6%    |  91.1%   |   80%    | 9000,4000  | 0.001,0.0001    | 100   | Adam       |  10       |   30     | 100       |  16000      |  0.3    |  0.9     |   40    |   speech_dataset         |    audio    |
|  7  | ziippy  | conv    |  86.8%   |  89.1%   |     72%   |   9000,3000 | 0.001,0.0001  | 100   | RMSProp    |  10       |   30     |  0        |  16000      |  0.1    |  0.8    |   40    |   speech_dataset_1200    |    audio_1200   |
|  8  | ziippy  | conv    |  85.5%   |  84.7%   |     75%   |   9000,3000 | 0.001,0.0001   | 100   | RMSProp    |  10       |   30     |  0        |  16000      |  0.1    |  0.8    |   40    |   speech_dataset_shift   |    audio        |
|  9  | joongjum| squeeze |  94.2%   |  95.5%   |   82 %    | 8000,4000  | 0.001,0.0001    | 100   | RMSProp    |  10       |   30     | 100       |  16000      |  0.1    |  0.8    |   40    |   speech_dataset         |    audio    |
|  10 | joongjum| squeeze |  92.3%   |  92.7%   |   82 %    | 9000,6000  | 0.001,0.0001    | 100   | Momentum    |  10       |   30     | 100       |  16000      |  0.2    |  0.8    |   40    |   speech_dataset_shift  |    audio    |
| 11  | hong  | mobilenet    |   95.0 %  |   94.7%   |   83%    | 9000,3000 |   0.002,0.0001  |   50  | Momentum  |   10   |   30     | 100       |  16000      |  0.3    |  0.8    |   40    |   speech_dataset         |   audio     |
| 12  | hong  | resnet    |   94.5 %  |   95.7%   |   83%    | 9000,3000 |   0.002,0.0001  |   50  | Momentum    |   10      |   30     | 100       |  16000      |  0.3    |  0.8    |   40    |   speech_dataset         |   audio     |      
|  13 | joongjum| squeeze |  95.1%   |  94.9%   |   84%    | 9000,6000  | 0.001,0.0001    | 100   | Adam    |  10       |   30     | 100       |  16000      |  0.2    |  0.8    |   40    |   speech_dataset_shift  |    audio    |
|  14 | hong| mobilenet |  95.2%   |  96.5%   |   85 %    | 9000,3000  | 0.003,0.0001    | 50   | RMSProp    |  10       |   30     | 100       |  16000      |  0.3    |  0.8    |   40    |   speech_dataset  |    audio    |
|  15 | joongjum| squeeze |  95.0%   |  94.6%   |   84%(better than No.13)    | 9000,6000  | 0.001,0.0001    | 100   | Adam    |  10       |   30     | 100       |  16000      |  0.3    |  0.8    |   40    |   speech_dataset_shift  |    audio    |
|  16 | joongjum| squeeze |  94.8%   |  95.0%   |   85%    | 9000,3000  | 0.001,0.0001    | 100   | Adam    |  10       |   30     | 100       |  16000      |  0.3    |  0.8    |   40    |   speech_dataset_shift  |    audio    |
|  17 | joongjum| squeeze |  94.3%   |  94.1%   |   86%    | 18000,6000  | 0.001,0.0001    | 100   | Adam    |  10       |   30     | 100       |  16000      |  0.3    |  0.8    |   40    |   speech_dataset_shift_gain_  |    audio    |
|  18 | joongjum| squeeze |  94.3%   |  94.6%   |   85%    | 18000,6000  | 0.001,0.0001    | 100   | Adam    |  10       |   30     | 100       |  16000      |  0.3    |  0.7    |   40    |   speech_dataset_shift_gain_  |    audio    |


## tips
- validation accuracy/loss 그래프 변화를 살펴보고 오버피팅 여부를 파악하여 epoch 값을 정한다.
- use tensorboard
- epoch count : 12000
- train batch size : 100
- prediction batch size : 200

