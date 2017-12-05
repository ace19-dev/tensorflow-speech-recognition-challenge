## What's the "Right" Neural Network for Use in a small device
- Sufficiently high accuracy (충분한 정확도를 가져야 함)
- Low computational complexity
- Low energy usage
- Small model size

## Why Small Deep Neural Networks?
- Small DNNs train faster on distributed hardware
- Small DNNs are more deployable on embedded processors
- Small DNNs are easily updatable Over-The-Air(OTA)
  - OTA 가 지원되었으면 원격으로 소프트웨어 업데이트를 하여 폭스바겐과 같은 대규모 리콜 상태를 막아서 손실을 줄일 수 있었을 거임.
  - 테슬라 같은 곳은 OTA 에 투자를 많이 하고 있음.

## Techniques for Small Deep Neural Networks
- Remove Fully-Connected Layers
  - CNN 기준, 파라미터의 90% 가 fc layer 에 들어감.
  - 파라미터 sharing 을 안하기 때문에 파라미터가 많다.(??)
- Kernel Reduction (filter 를 3x3 -> 1x1 로 줄여서 연산량이나 파라미터 수를 줄인다.)
  - 대표적인 것이 스퀴즈 넷
- <b>Channel Reduction</b>
- Evenly Spaced Down-sampling
  - 초반에 down-sampling 을 하면 네트웍 사이즈는 줄어들지만 정확도가 많이 떨어지게 되고,
  - 후반에 down-sampling 을 하게 되면 정확도는 괜찮지만, 연산량과 파라미터 수가 많아진다.
  - 중간에 잘 절충해서 VGG 같이 down-sampling 을 네트웍 전체에 골고루 퍼지게 하면 적당하게 만들 수 있다.
- <b>Depthwise Separable Convolutions</b>
- Shuffle Operations
- <b>Distillation & Compression</b>

## key Idea
- Depthwise Separable Convolutions
  - 성능도 괜찮고 슬림한 네트웍을 할 수 있다.

## Recap
- 일반적으로 2D conv 라고 말하지만, 실제로는 3D 오퍼레이션이 들어간다. (w, h, c)
- 예를 들어 3x3 필터를 사용한다고 하면 채널수는 언급되지 않지만, 항상 필터의 채널수는 입력의 채널수와 같다.
- 아웃풋 채널수는 필터를 몇 개 사용하는냐에 달렸다.
- conv 는 w, h, c 를 모두 동시에 한꺼번에 고려해서 그것을 다 element-wise 곱을 하고 전체 sum 을 해서 하나의 숫자로 표현한다.
- VGG 에서는 3x3 필터만 사용한다. 그전에는 어떤 사이즈의 필터를 사용해야 하는지 고민이 많았다.
- 3x3 필터를 여러번 사용하면 5x5 또는 7x7 conv layer 와 같은 effective receptive field 를 가진다.
- more non-linearities, fewer parameters -> regularization effect

## To be continue...


## reference
- https://www.youtube.com/watch?v=7UoOFKcyIvM&index=45&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS










