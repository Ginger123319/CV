from itertools import islice
from multiprocessing import Pool
import multiprocessing
import os
from pathlib import Path
import cv2
import time
import json
import cProfile
import pstats
import subprocess
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from TransNetV2.inference.transnetv2 import TransNetV2

from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode

from optical_flow import extract_clip_frames

# 深度学习方法预测剪切点


def predict_video(video_path, model, fps, gpu_id):

    with tf.device(f'/GPU:{gpu_id}'):
        video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
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


def convert2frameTimeCode(cut_points, frame_rate):
    timecodes = []
    for start, end in cut_points:
        start_timecode = FrameTimecode(start / frame_rate, fps=frame_rate)
        end_timecode = FrameTimecode(end / frame_rate, fps=frame_rate)
        timecodes.append((start_timecode, end_timecode))
    for index, (start_tc, end_tc) in enumerate(timecodes):
        print(f"Cut point {index + 1}: Start - {start_tc}, End - {end_tc}\n")
    return timecodes

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


def split2scenes(video_path, output_dir, model, gpu_id):
    if not has_video_stream(video_path):
        print(f"Error: {video_path} has no video stream.")
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: {video_path} is not a video file.")
        cap.release()
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    scene_list, video_frames = predict_video(video_path, model, fps, gpu_id)
    scene_list = extract_clip_frames(video_frames, scene_list, fps)
    if scene_list:
        scene_list = convert2frameTimeCode(scene_list, fps)
        split_video_ffmpeg(video_path, scene_list,
                           show_progress=True, output_dir=output_dir)
    else:
        print(f"No scene detected in {video_path}.")

def run_inference_on_gpu(gpu_id, output_dir, video_files):
    # 将CUDA_VISIBLE_DEVICES环境变量设置为特定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f'GPU ID is {gpu_id}:\n')
    print(video_files)
    
    with tf.device(f'/GPU:{gpu_id}'):
        model = TransNetV2()

    # 针对IO（读写磁盘频繁）密集型任务，进行多线程处理

    # 线程数设置为服务器cpu核心数的一半
    with ThreadPoolExecutor(max_workers=48) as executor:
        executor.map(lambda path: split2scenes(
            path, output_dir, model, gpu_id), video_files)
    # for path in video_files:
    #     split2scenes(path, output_dir, model, gpu_id)

def batch_iterator(iterator, batch_size):
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

if __name__ == "__main__":
    # 定义两个GPU ID
    gpu_ids = [0, 1]

    directory = "input_dir"

    # 常用的视频格式后缀
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = (f for f in Path(directory).rglob('*') if f.is_file() and f.suffix in video_extensions)

    # 获取给定路径的父级目录
    parent_directory = os.path.dirname(directory)
    output_dir = os.path.join(parent_directory, "output_clips")

    batch_size = 6
    video_files = batch_iterator(video_files, batch_size)

     # 创建两个进程，每个进程使用一个GPU
    p1 = multiprocessing.Process(target=run_inference_on_gpu, args=(0, output_dir, next(video_files)))
    p2 = multiprocessing.Process(target=run_inference_on_gpu, args=(1, output_dir, next(video_files)))
    
    # 启动进程
    p1.start()
    p2.start()
    
    # 等待进程结束
    p1.join()
    p2.join()