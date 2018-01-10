import os
import wave, struct
import numpy as np
import array
import matplotlib.pyplot as plt
from shutil import copyfile
from pydub import AudioSegment

########################################################################################## Utility ################
global global_figure_no

# silent 인지 여부 체크
def is_silent(threshold, peak_value):
    if pow(threshold, 2) >= pow(peak_value, 2):
        return True
    return False

def get_sound(file):
    # AudioSegment
    sound = AudioSegment.from_file(file, "wav")
    return sound

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def get_normalized_sound(file, normalization_target_amp):
    # AudioSegment
    sound = get_sound(file)
    normalized_sound = match_target_amplitude(sound, normalization_target_amp)
    return sound

######################################################################################## np.xxx_shift 기반 #######

# np.xxx_shift 함수를 이용해서 wav sample 값을 변경 한 후 to_file 로 저장하는 함수
def export_wav_using_shift(samples, sound, is_right_shift, shift_val, to_file):
    if is_right_shift:
        shifted_samples = np.right_shift(samples, shift_val)
    else:
        shifted_samples = np.left_shift(samples, shift_val)

    # now you have to convert back to an array.array
    shifted_samples_array = array.array(sound.array_type, shifted_samples)
    new_sound = sound._spawn(shifted_samples_array)
    new_sound.export(to_file, format="wav")
    return

# from_file 을 읽고 silent 가 아니면 export_wav_using_shift 함수 호출
def wav_generator_using_shift(do_print, from_file_name, to_file_name, threshold):
    from_file = from_file_name + ".wav"

    sound = get_sound(from_file)
    peak_amplitude = sound.max
    if do_print:
        print("peak_amplitude: ", peak_amplitude)

    # Step 1 - peak_amplitude 값이 임계치보다 작으면 해당 파일은 silent 라고 판단하고 return False
    if is_silent(threshold, peak_amplitude):
        print("peak_amplitude is lower than threshold (%s) (%s < %s)" % (from_file_name, peak_amplitude, threshold))
        return False

    # Step 2 - samples 값에 대해 left_shift, right_shift 를 이용하여 변형 파일 생성
    samples = sound.get_array_of_samples()
    # print(samples)

    # shift option
    shift_from_val = 1
    shift_to_val = 2
    using_left_shift = True
    using_right_shift = True

    # right shift
    if using_right_shift:
        for i in range(shift_from_val, shift_to_val+1):
            is_right_shift = True
            to_file = to_file_name + "_right" + str(i) + ".wav"
            export_wav_using_shift(samples, sound, is_right_shift, i, to_file)

    # left shift
    if using_left_shift:
        for i in range(shift_from_val, shift_to_val+1):
            is_right_shift = False
            to_file = to_file_name + "_left" + str(i) + ".wav"
            export_wav_using_shift(samples, sound, is_right_shift, i, to_file)

    return True

############################################################################# AudioSegment.apply_gain 기반 #######

# AudioSegment.apply_gain 함수를 이용해서 생성된 sound 객체를 이용하여 to_file 로 저장하는 함수
def export_wav_using_gain(sound, value, to_file):
    if value != 0:  # 0 일 때는 제외 - 동일 gain 이므로
        # make sound change by value - (value > 0 then louder, value < 0 then quieter)
        new_sound = sound.apply_gain(value)
        new_sound.export(to_file, format="wav")
    return

# from_file 을 읽고 silent 가 아니면 export_wav_using_gain 함수 호출
def wav_generator_using_gain(do_print, from_file_name, to_file_name, threshold, max_gain):
    from_file = from_file_name + ".wav"

    sound = get_sound(from_file)
    peak_amplitude = sound.max
    if do_print:
        print("peak_amplitude: ", peak_amplitude)

    # Step 1 - peak_amplitude 값이 임계치보다 작으면 해당 파일은 silent 라고 판단하고 return False
    if is_silent(threshold, peak_amplitude):
        print("peak_amplitude is lower than threshold (%s) (%s < %s)" % (from_file_name, peak_amplitude, threshold))
        return False

    # Step 2 - gain 을 이용한 volume change
    for i in range(-max_gain, max_gain+1):
        to_file = to_file_name + "_vol_" + str(i) + ".wav"
        export_wav_using_gain(sound, i, to_file)

    return True

################################################################################## start_position 기반 ##############

def print_figure(figure_no, x_val, y_val, x_label, y_label):
    plt.subplot(figure_no)
    plt.plot(x_val, y_val, linewidth=0.1, color='#000000')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return figure_no+1

# samples 에서 start position 부터 새로운 to_file 로 저장하는 함수
def export_wav_from_position(sound, samples, start_position, append_silent, figure_use, figure_no, figure_rate, to_file):
    new_samples = []
    if start_position > 0:      # 왼쪽으로 민다
        # 먼저 파형을 복사
        for i in range(start_position, len(samples)):
            new_samples.append(samples[i])

        # 이후 append silent
        if append_silent == True:
            for i in range(0, start_position):
                new_samples.append(0)
    else:
        # 먼저 append silent
        if append_silent == True:
            for i in range(0, -start_position):
                new_samples.append(0)

        # 이후 파형을 append
        for i in range(0, len(samples)+start_position):
            new_samples.append(samples[i])

    # figure - new
    if figure_use == True:
        # plot amplitude (or loudness) over time
        time = np.arange(0, len(new_samples), 1) / figure_rate
        global_figure_no = print_figure(figure_no, time, new_samples, "Time (s)", "Amplitude")

    # now you have to convert back to an array.array
    new_samples_array = array.array(sound.array_type, new_samples)
    new_sound = sound._spawn(new_samples_array)
    new_sound.export(to_file, format="wav")
    return

# from_file 을 읽고 silent 가 아니면, start position 을 찾은 후 export_wav_from_position 함수 호출
def wav_generator_using_posotion(do_print, from_file_name, to_file_name, threshold, append_silent, need_to_figure, need_to_normalization, normalization_target_amp):
    from_file = from_file_name + ".wav"

    # wav volume normalization
    if need_to_normalization == True:
        normalized_file = from_file + ".normalized.wav"
        sound = get_normalized_sound(from_file, normalization_target_amp)
        sound.export(normalized_file, format="wav")
        from_file = normalized_file

    '''
    # Open    
    w = wave.open(from_file, "rb")
    p = w.getparams()
    rate = p[2]
    f = p[3]  # number of frames
    s = w.readframes(f)
    w.close()

    # Debug
    if do_print == True:
        print("Number of Total Frames: ", f)
    '''

    # figure
    if need_to_figure == True:
        plt.figure(1)

    ##### [0] variables
    global_figure_no = 511

    ##### [1] open
    sound = get_sound(from_file)
    samples = sound.get_array_of_samples()
    rate = sound.frame_rate
    total_frames = len(samples)
    time = np.arange(0, total_frames, 1) / rate
    frame_per_100 = int(rate / 10)             # 0.1초에 대한 데이터의 개수

    # figure - org
    if need_to_figure == True:
        # plot amplitude (or loudness) over time
        global_figure_no = print_figure(global_figure_no, time, samples, "Time (s)", "Amplitude")

    ##### [2] Find start position over than threshold
    ## start position 은 찾지 않기로 함 - 18.01.10
    '''
    start_position = 0
    invalid_frame_count = 0
    process_frame_count = 0
    stop_after_first_start_position = False      # start position 을 찾은 후 바로 멈출 것인지 여부
    for i in range(0, len(samples)):
        # print("Amplitude[%s]: %s" %  (i, s[i]))

        if pow(samples[i], 2) >= pow(threshold, 2):
            # 임계치 이외 값이 들어오면 start_position 변수 값 설정
            if start_position == 0:
                # Debug
                if do_print == True:
                    print("Find start pos: ", i, ",  ", samples[i], ",  ", threshold)
                start_position = i
                if stop_after_first_start_position:     # 무조건 처음에 찾으면 멈춰?
                    break
        else:
            # 임계치 이내 값이 들어오면 invalid_frame_count 값을 증가시킨다.
            invalid_frame_count += 1

            # 만약 invalid_frame_count 값이 0.1 초 동안에 대한 frame count 보다 크거나 같으면 start_position 을 다시 구한다.
            if invalid_frame_count >= frame_per_100:
                # Debug
                if do_print == True:
                    print("Invalid frame count over than 100ms: ", samples[i], ", ======> ", invalid_frame_count)
                start_position = 0
                invalid_frame_count = 0
                process_frame_count = 0
                continue

        # 처리한 프레임의 개수를 증가시키고, 이 값이 0.11 초 동안에 대한 frame count 보다 크면 start_position 을 찾는 것을 멈춘다
        process_frame_count += 1
        if process_frame_count >= (frame_per_100*1.1):
            # Debug
            if do_print == True:
                print("Finally Find the start pos: ", start_position)
            break

    # Debug
    if do_print == True:
        print(">> start position is : ", start_position)

    ##### [3] create new wave
    if start_position > 0:

        # 1개만 생성 시
        #to_file = to_file_name + ".wav"
        #export_wav_from_position(sound, samples, start_position, append_silent, need_to_figure, global_figure_no, rate, to_file)

        # 여러 개 생성
        gen_no = 1
        gap_value = int(frame_per_100 / 2)  # 0.05 초 씩?
        real_start_position = gap_value
        #while start_position > 0:
        while real_start_position < start_position:
            to_file =  to_file_name + "_gen_" + str(gen_no) + ".wav"
            print(to_file)
            #global_figure_no = export_wav_from_position(sound, samples, start_position, append_silent, need_to_figure, global_figure_no, rate, to_file)
            export_wav_from_position(sound, samples, real_start_position, append_silent, need_to_figure, global_figure_no, rate, to_file)
            gen_no += 1
            real_start_position += gap_value
    '''

    # 좌/우로 이동해서 총 10개 생성
    # 왼쪽에서 5번
    gen_no = 1
    gap_value = int(frame_per_100 / 2)  # 0.05 초
    real_start_position = gap_value
    # while start_position > 0:
    while gen_no <= 5:
        to_file = to_file_name + "_gen_left_" + str(gen_no) + ".wav"
        #print(to_file)
        # global_figure_no = export_wav_from_position(sound, samples, start_position, append_silent, need_to_figure, global_figure_no, rate, to_file)
        export_wav_from_position(sound, samples, real_start_position, append_silent, need_to_figure, global_figure_no,
                                 rate, to_file)
        gen_no += 1
        real_start_position += gap_value

    # 오른쪽에서 5번
    gen_no = 1
    gap_value = int(frame_per_100 / 2)  # 0.05 초
    real_start_position = -gap_value
    # while start_position > 0:
    while gen_no <= 5:
        to_file = to_file_name + "_gen_right_" + str(gen_no) + ".wav"
        #print(to_file)
        # global_figure_no = export_wav_from_position(sound, samples, start_position, append_silent, need_to_figure, global_figure_no, rate, to_file)
        export_wav_from_position(sound, samples, real_start_position, append_silent, need_to_figure,
                                 global_figure_no,
                                 rate, to_file)
        gen_no += 1
        real_start_position -= gap_value


    '''
    new_wav = []
    for i in range(start, len(s)):
        new_wav.append(s[i])

    # add silent
    if add_silent == True:
        for i in range(0, start):
            new_wav.append(0)

    # convert amplitude value to frame data
    time2 = np.arange(0, len(new_wav), 1) / rate

    # figure
    if need_to_figure == True:
        # plot amplitude (or loudness) over time
        plt.subplot(212)
        plt.plot(time2, new_wav, linewidth=0.1, color='#000000')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    s2 = struct.pack('h' * len(new_wav), *new_wav)

    # Save
    if start > 0:
        w = wave.open(to_file, "wb")
        w.setparams(p)
        w.writeframes(s2)
        w.close()
    else:
        copyfile(from_file, to_file)
        result = False

    # wav volume normalization
    if need_to_normalization == True:
        os.remove(from_file)
    '''

    # figure
    if need_to_figure == True:
        plt.show()

    return True

def main():
    do_print = True
    wav_volume_threshold = 1200
    do_append_silent = True
    do_figure = True
    do_wav_volume_normalization = False
    wav_volume_normalization_target = -30.0

    from_file_name = "00f0204f_nohash_0"
    to_file_name = from_file_name + "____"

    wav_generator_using_posotion(do_print, from_file_name, to_file_name, wav_volume_threshold, do_append_silent, do_figure, do_wav_volume_normalization, wav_volume_normalization_target)

    #result = wav_generator_using_shift(do_print, from_file_name, to_file_name, wav_volume_threshold)

    max_volume_gain = 20
    #result = wav_generator_using_gain(do_print, from_file_name, to_file_name, wav_volume_threshold, max_volume_gain)

    #if result == True:
        # 원본 copy
        #copyfile(from_file_name+".wav", to_file_name+".wav")

if __name__ == '__main__':
    main()


