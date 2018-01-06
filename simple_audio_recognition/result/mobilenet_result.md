

### Results
| Model | Accuracy | Traning Step   | Learning Rate       | Batch Size | Optimizer                | Activation function | silence_percentage | unknown_percentage | time_shift_ms | sample_rate |
|-------|----------|----------------|---------------------|------------|--------------------------|---------------------|--------------------|--------------------|---------------|-------------|
| M1    | 86.5%    | 15000/3000     | 0.01/0.001/0.0001   | 100        | GradientDescentOptimizer | Relu                | 10                 | 10                 | 100           | 16000       |
| M2    | 89.2%    | 15000/3000     | 0.01/0.001/0.0001   | 100        | GradientDescentOptimizer | Relu                | 10                 | 10                 | 100           | 16000       |
| M2    | 93.8%    | 8000/5000/3000 | 0.01/0.002/0.0001   | 100        | RMSPropOptimizer         | Relu                | 10                 | 10                 | 100           | 16000       |
| M2    | 94.2%    | 7000/5000/4000 | 0.008/0.0005/0.0001 | 100        | RMSPropOptimizer         | Relu                | 10                 | 10                 | 150           | 16000       |
| M2    | 94.8%    | 7000/5000/4000 | 0.007/0.0004/0.0001 | 100        | RMSPropOptimizer         | Relu                | 10                 | 10                 | 150           | 16000       |
| M3    | 95.7%    | 4000/5000/7000 | 0.005/0.0002/0.0001 | 100        | RMSPropOptimizer         | Relu                | 10                 | 10                 | 150           | 16000       |

### Model features
| Network Name    | Architecture                                                          | Filter </br> (H, W, C, N)                                                                      | Feature Size (height, width, channels)                                                                            | Memory Usage |
  |-----------------|-----------------------------------------------------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|--------------|
  | M1              | Conv / s2 </br> Conv dw / s1 </br> Conv / s1 </br> Avg Pool / s1                        | 3 x 3 x 1 x 32  </br> 3 x 3 x 32 x 32 </br> 1 x 1 x 32 x 64                                  | 65 x 40 x 1 </br> 32 x 19 x 32 </br> 32 x 19 x 32 </br> 32 x 19 x 64                                       | 80.6K        |
  | M2              | Conv / s2 </br> Conv dw / s1 </br> Conv / s1 </br> Conv dw / s2 </br> Conv / s1  </br> Avg Pool / s1 | 3 x 3 x 1 x 32  </br> 3 x 3 x 32 x 32 </br> 1 x 1 x 32 x 64</br>  3 x 3 x 64 x 64 </br> 1 x 1 x 64 x 128 | 65 x 40 x 1 </br> 32 x 19 x 32 </br> 32 x 19 x 32 </br> 32 x 19 x 64  </br> 15 x 9 x 64 </br> 15 x 9 x 128 | 123.6K       |
  | M3              |  |  |  | 130.5K       |
|                 |                                                                       |                                                                                  |                                                                                          |              |
 | cnn-trad-fpool3 | Conv2d </br> MaxPool </br> Conv2d                                                 |                                                                                  | 65 x 40 x 64 </br> 33 x 20 x 64 </br> 33 x 20 x 64                                                   | 250.9K       |


### Tensorboard
![alt text](https://i.imgur.com/eqadZIy.png)
