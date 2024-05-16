import math
import os
from pathlib import Path
import cv2
import time
import json
import cProfile
import pstats
import subprocess
from concurrent.futures import ThreadPoolExecutor

from TransNetV2.inference.transnetv2 import TransNetV2

from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode

from optical_flow_copy import extract_clip_frames

from multiprocessing import Lock

# 定义全局变量
model = None
model_lock = Lock()
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']


# 深度学习方法预测剪切点
def predict_video(video_path, model, fps):

    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(
        video_path)
    scene_list = model.predictions_to_scenes(single_frame_predictions)

    # 计算需丢弃的帧数
    frames_to_drop = int(fps)

    # 过滤并调整场景列表
    new_ranges = [
        (start + frames_to_drop, end - frames_to_drop)
        for start, end in scene_list
        if end - start > 4 * frames_to_drop  # 确保场景长度足够
    ]

    return new_ranges, video_frames

# 将帧号转换为FrameTimecode


def convert2frameTimeCode(file_name, cut_points, frame_rate):
    timecodes = []
    clip_names = []
    timecodes_serial = []
    total_scenes = len(cut_points)
    file_name = os.path.splitext(file_name)[0]
    for index, (start, end) in enumerate(cut_points):
        scene_num = (
            ('%0' + str(max(3, math.floor(math.log(total_scenes, 10)) + 1)) + 'd') % (index + 1))
        clip_name = f"{file_name}-Scene-{scene_num}.mp4"
        clip_names.append(clip_name)
        start_timecode = FrameTimecode(start / frame_rate, fps=frame_rate)
        end_timecode = FrameTimecode(end / frame_rate, fps=frame_rate)
        timecodes.append((start_timecode, end_timecode))
        timecodes_serial.append(
            (start_timecode.get_timecode(), end_timecode.get_timecode()))
    # for index, (start_tc, end_tc) in enumerate(timecodes):
    #     print(f"Cut point {index + 1}: Start - {start_tc}, End - {end_tc}\n")
    return timecodes, timecodes_serial, clip_names

# 判断是否有视频流


def has_video_stream(filename):
    try:
        cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams {filename}'
        output = subprocess.check_output(cmd, shell=True, text=True)
        data = json.loads(output)
        return any(stream['codec_type'] == 'video' for stream in data['streams'])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while checking video stream: {e}")
        return False

# 初始化模型


def get_model():
    global model
    if model is None:
        with model_lock:
            if model is None:  # 再次检查模型是否已初始化，这是必要的，因为锁释放后，其他进程可能已经初始化了模型
                model = TransNetV2()
    return model

# 分割视频主方法


def split2scenes(video_path, output_dir):
    if not os.path.isfile(video_path):
        print(f"Error: {video_path} is not a file.")
        return None
    else:
        file_name = os.path.basename(video_path)
        file_extension = os.path.splitext(file_name)[1]
        if file_extension not in video_extensions:
            print(f"Error: {file_extension} is not a video file extension.")
            return None

    if not has_video_stream(video_path):
        print(f"Error: {video_path} has no video stream.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path} cannot be opened.")
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    model = get_model()
    scene_list, video_frames = predict_video(video_path, model, fps)
    scene_list, motion_score_list = extract_clip_frames(
        video_frames, scene_list, fps)
    if scene_list:
        result_dict = {}
        scene_list, timecode_serial, clip_names = convert2frameTimeCode(
            file_name, scene_list, fps)
        result_dict['video_path'] = video_path
        result_dict['scene_list'] = timecode_serial
        result_dict['output_dir'] = output_dir
        result_dict['clip_names'] = clip_names
        result_dict['motion_score_list'] = motion_score_list
        split_video_ffmpeg(video_path, scene_list,
                           show_progress=True, output_dir=output_dir)
        result = json.dumps(result_dict, indent=4)
        return result
    else:
        print(f"No scene detected in {video_path}.\n")
        return None


if __name__ == '__main__':

    directory = "input_dir"

    # 常用的视频格式后缀
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = (str(f) for f in Path(directory).rglob(
        '*') if f.is_file() and f.suffix in video_extensions)

    # 获取给定路径的父级目录
    parent_directory = os.path.dirname(directory)
    output_dir = os.path.join(parent_directory, "output_clips")

    # 针对IO（读写磁盘频繁）密集型任务，进行多线程处理
    start_time = time.time()
    profile = cProfile.Profile()
    profile.enable()

    # 线程数设置为服务器cpu核心数的一半
    with ThreadPoolExecutor(max_workers=48) as executor:
        executor.map(lambda path: split2scenes(
            path, output_dir), video_files)
    for path in video_files:
        result = split2scenes(path, output_dir)
        # print(result)
        # break

    profile.disable()
    stats = pstats.Stats(profile)
    stats.sort_stats("cumulative").print_stats(10)  # 打印前10行数据
    print(f"Total time: {time.time() - start_time} seconds")
