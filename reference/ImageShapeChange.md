## Image Shape change
- height
  - new_train.py
    - dct_coefficient_count value change (변경을 원하는 값으로)
    - sample_rate
  - new_input_data.py
    - prepare_processing_graph method 제일 하단부에 있는 self.mfcc = contrib_audio.mfcc에 parameter로 upper_frequency_limit, filterbank_channel_count 값을 추가
    - filterbank_channel_count 값은 dct_coefficient_count 값과 같은 값으로 할당
    - upper_frequency_limit 값은 dct_coefficient_count 값이 커지는 것과 비례해서 크기를 늘림
    
- width
  - new_train.py
    - Formula : width = (clip_duration_ms-window_size_ms)/window_stride_ms 
    
- TODO
  - 위의 계산방식의 의미 정리, sample_rate, upper_frequency_limit 정확한 값 계산 방법 
  - parameter 변경 없이 얻어진 행렬 값을 resize 하는 방식으로 변경
