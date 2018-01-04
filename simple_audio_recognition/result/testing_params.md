
### tries
|  No | Testor| Valid Acc.|Test Acc.| Pred Acc.| epoch     | Learning Rate   | Batch | Optimizer       | Activation |win_stride|win_size| time_shift|sample_rate |back_vol.|back_freq.|dct_coef.|
|-----|-------|----------|----------|----------|-----------|-----------------|-------|-----------------|------------|----------|--------|-----------|-------------|--------|----------|---------|
|  1  | Sean  |  73.8%   |  72.5%   |          |2000,3000  | 0.0003,0.00007  | 100   | SDG             | Relu       | 10       |   30   | 100       |  16000      |  0.5   |  0.8     |   40    | 
|  2  |       |          |          |          |           |                 |       |                 |            |          |         |         |              |        |           |        |
|  3  |       |          |          |          |           |                 |       |                 |         |            |          |         |              |        |           |        |
|  4  |       |          |          |          |           |                 |       |                 |         |            |          |         |              |        |           |        |
|  5  |       |          |          |          |           |                 |       |                 |         |            |          |         |              |        |           |        |
|  6  |       |          |          |          |           |                 |       |                 |         |            |          |         |              |        |           |        |



### cautions
- validation accuracy/loss 그래프 변화를 살펴보고 오버피팅 여부를 파악하여 epoch 값을 정한다.
- tensorboard 를 이용한다.