## What's the "Right" Neural Network for Use in a small device
- Sufficiently high accuracy (충분한 정확도를 가져야 함)
- Low computational complexity
- Low energy usage
- Small model size

## Why Small Deep Neural Networks?
- Small DNNs train faster on distributed hardware
- Small DNNs are more deployable on embedded processors
- Small DNNs are easily updatable Over-The-Air(OTA)
  - 소프트웨어 업데이트를 통신망을 이용해서 하겠다. 라는 개념
  - OTA 가 지원되었으면 폭스바겐과 같은 대규모 리콜 상태를 막아서 손실을 줄일 수 있었을 거임.
  - 테슬라, GM 같은 곳은 OTA 에 투자를 많이 하고 있음.
  - 딥러닝 모델을 사용할 때 파라미터들을 새로 retraining 해서 업데이트 할때 사이즈가 너무 크면 문제가 된다.
  - 점점 Small Deep Neural Network 의 중요성이 부각될 것이다.

## Techniques for Small Deep Neural Networks
- Remove Fully-Connected Layers
  - CNN 기준으로 보면 파라미터의 90% 가 Fully-Connected Layers 에 들어감.
  - 파라미터 sharing 을 안하기 때문에 파라미터가 많다.(??)
- Kernel Reduction (filter 를 3x3 -> 1x1 로 줄여서 연산량이나 파라미터 수를 줄인다.)
  - 대표적인 것이 SqueezeNet
- <b>Channel Reduction</b>
- Evenly Spaced Down-sampling
  - 초반에 down-sampling 을 하면 네트웍 사이즈는 줄어들지만 정확도가 많이 떨어지게 되고,
  - 후반에 down-sampling 을 하게 되면 정확도는 괜찮지만, 연산량과 파라미터 수가 많아진다.
  - 중간에 잘 절충해서 VGG 같이 down-sampling 을 네트웍 전체에 골고루 퍼지게 하면 정확도를 적당하게 만들면서 slim 하게 할 수 있다.
- <b>Depthwise Separable Convolutions</b>
- Shuffle Operations
- <b>Distillation & Compression</b>

## key Idea
- 이 논문의 kep idea 는 <b>Depthwise Separable Convolutions</b> 을 사용하겠다.
  - 성능도 괜찮고 슬림한 네트웍을 만들 수 있다.

## CNN Recap
- 일반적으로 2D conv. 라고 말하지만, 실제로는 3D operation 이 들어간다. (w, h, c)
- 예를 들어 3x3 필터를 사용한다고 하면 채널수는 언급되지 않지만, 항상 필터의 채널수는 입력의 채널수와 같다.
- <b>아웃풋 채널수는 사용하는 필터의 개수가 된다.</b>
- conv. 는 w, h, c 를 모두 동시에 한꺼번에 고려해서 그것을 다 element-wise 곱을 하고 전체를 sum 을 해서 하나의 숫자로 표현한다.
- VGG 에서는 3x3 필터만 사용한다. 그전에는 어떤 사이즈의 필터를 사용해야 할지 고민이 많았다.
- 3x3 필터를 여러번 사용하면 5x5 또는 7x7 conv. layer 와 같은 effective receptive field (http://laonple.blog.me/220594258301) 를 가진다.
- more non-linearities, fewer parameters -> regularization effect
- Why should we always consider all channels?

## Depthwise Separable Convolution
- 기존 conv. 는 채널을 전부 고려해서 연산을 하지만, 여기에선 한 채널씩 떼어서 연산.
- Depthwise Conv. + Pointwise Conv.(1x1 Conv.)
- 채널 방향과 w/h 방향을 분리해서 연산함.
- 만약 3x3 depthwise separable convolutions 적용하면 8~9 배의 연산량을 줄일 수 있다.

## Additional idea - Width Multiplier / Resolution Multiplier
- Width Multiplier: Conv. 필터(채널)수를 적게 만들고 돌려보겠다 라는 의미
- Resolution Multiplier: 입력 이미지의 가로/세로를 줄여서 입력하겠다 라는 의미

## implementation
- tensorflow 공식 github 의 models 에 slim 모델 참고
- slim.seperable_convolution2d

## reference
- https://www.youtube.com/watch?v=7UoOFKcyIvM&index=45&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS










