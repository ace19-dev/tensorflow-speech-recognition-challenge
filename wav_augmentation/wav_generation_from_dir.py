import os
import sys
import wave, struct
import numpy as np
import matplotlib.pyplot as plt
from shutil import *

from wav_processor import *

def check_dir(dir):
    # create directory is not exists
    if not os.path.exists(dir):
        os.makedirs(dir)

def remove_dir(dir):
    if os.path.exists(dir):
        rmtree(dir)

def count_files(dir):
    x = 0
    for root, dirs, files in os.walk(dir):
        for f in files:
            x = x + 1
    return x

def main():
    do_print = False
    wav_volume_threshold = 1200
    do_append_silent = True
    do_figure = False
    do_wav_volume_normalization = False
    wav_volume_normalization_target = -30.0

    # 동작 모드
    using_startpoint = True
    using_gain = True
    volume_gain_min = -35
    volume_gain_max = -19
    volume_gain_step = 2

    basedir = "D:\\tmp"
    dataset_dir = "speech_dataset"
    target_dir_prefix = "_"
    #if do_wav_volume_normalization:
    #    target_dir_prefix += ("_volume_" + str(wav_volume_normalization_target) + "_")

    if using_startpoint:
        target_dir_prefix += ("timeshift_")
    if using_gain:
        target_dir_prefix += ("gain_10x_")

    target_dir_prefix = dataset_dir + target_dir_prefix

    rootdir = basedir + "\\" + dataset_dir
    targetdir = basedir + "\\" + target_dir_prefix
    #remove_dir(targetdir)

    print("root directory prefix: " + rootdir)
    print("target directory prefix: " + targetdir)

    result = True
    process_count = 0
    total_file_count = count_files(rootdir)
    for root, subdirs, files in os.walk(rootdir):
        #for subdir in subdirs:
        #   print('\t- subdirectory ' + subdir)

        for filename in files:
            process_count += 1
            full_path = os.path.join(root, filename)
            target_path = root.replace(dataset_dir, target_dir_prefix)
            check_dir(target_path)

            target_full_path = os.path.join(target_path, filename)
            print('\t- %s (full path: %s -> target full path: %s) (%s/%s)' % (filename, full_path, target_full_path, process_count, total_file_count))

            # 예외 파일 처리
            if full_path.find(".wav") == -1:
                print("%s is not wav file" % full_path)
                copyfile(full_path, target_full_path)
                continue

            # _background_noise_ 폴더 제외
            if full_path.find("_background_noise_") != -1:
                print("%s is excepted file" % full_path)
                copyfile(full_path, target_full_path)
                continue

            if os.path.exists(full_path):
                with open(full_path, 'rb') as f:
                    # 확장자 제거
                    full_path = os.path.splitext(full_path)[0]
                    target_full_path = os.path.splitext(target_full_path)[0]

                    if using_startpoint:
                        result = wav_generator_using_posotion(do_print, full_path, target_full_path, wav_volume_threshold, do_append_silent, do_figure, do_wav_volume_normalization, wav_volume_normalization_target)

                    if result == True:
                        # 원본 copy
                        copyfile(full_path + ".wav", target_full_path + ".wav")

                    if result == True and using_gain:
                        gen_file_name = []
                        gen_file_name.append(target_full_path)
                        # 왼쪽으로 5번
                        for i in range(1, 5 + 1):
                            gen_file_name.append(target_full_path + "_gen_left_" + str(i))
                        # 오른쪽으로 4번
                        for i in range(1, 4 + 1):
                            gen_file_name.append(target_full_path + "_gen_right_" + str(i))

                        for i in range(0, len(gen_file_name)):
                            from_file_name = gen_file_name[i]
                            to_file_name = from_file_name + "___"
                            result = wav_generator_using_gain(do_print, from_file_name, to_file_name,
                                                              wav_volume_threshold, volume_gain_min, volume_gain_max,
                                                              volume_gain_step, do_figure)

            if result == False:
                print("wav_split return False (%s)" % (target_full_path))
                #break;

        #if result == False:
        #    break;

if __name__ == '__main__':
    main()