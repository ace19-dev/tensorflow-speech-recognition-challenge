## tried
|  No | Testor  | Network | Valid Acc.|Test Acc.| Pred Acc.| epoch     | Learning Rate   | Batch | Optimizer  | Activation |win_stride|win_size| time_shift|sample_rate |back_vol.|back_freq.|dct_coef.|
|-----|---------|---------|----------|----------|----------|-----------|-----------------|-------|------------|---------|----------|----------|----------|-------------|--------|----------|---------|
|  1  | Sean    |  conv   |  92.3%   |   91.6%  |   82%    | 9000,3000 | 0.001,0.0001    | 100   | Momentum   | Relu    | 10       |   30     | 100      |  16000      |  0.3   |  0.8     |   40    |
|  2  | Sean    |  mobile |  ?%      |   ?%     |   ?%     | 15000,3000| 0.001,0.0001    | 100   | RMSProp    | Relu    | 10       |   30     | 100      |  16000      |  0.3   |  0.9     |   40    |
|  3  |         |         |          |          |          |           |                 |       |            |         |          |          |          |  16000      |        |          |         |
|  4  |         |         |          |          |          |           |                 |       |            |         |          |          |          |  16000      |        |          |         |
|  5  |         |         |          |          |          |           |                 |       |            |         |          |          |          |  16000      |        |          |         |
|  6  |         |         |          |          |          |           |                 |       |            |         |          |          |          |  16000      |        |          |         |


## tips
- validation accuracy/loss 그래프 변화를 살펴보고 오버피팅 여부를 파악하여 epoch 값을 정한다.
- tensorboard 를 이용한다.
