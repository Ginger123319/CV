import cv2
import ffmpeg
import numpy as np

import subprocess
import json


def has_video_stream(filename):
    cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams {filename}'
    output = subprocess.check_output(cmd, shell=True)
    data = json.loads(output)
    for stream in data['streams']:
        if stream['codec_type'] == 'video':
            return True
    return False

# 按照指定帧率（2fps）抽取视频帧
def extract_frames(video_path, fps=2):

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    target_fps = min(frame_rate, fps)

    # 获取视频的分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_stream, err = ffmpeg.input(video_path).output(
        "pipe:", format="rawvideo", pix_fmt="rgb24",
    ).run(capture_stdout=True, capture_stderr=True)

    video_frames = np.frombuffer(
        video_stream, np.uint8).reshape([-1, height, width, 3])

    # frames = []
    # frame_count = 0
    # for frame in video_frames:
    #     if frame_count % int(frame_rate / target_fps) == 0:
    #         frames.append(frame)
    #     frame_count += 1

    frames = video_frames[::int(frame_rate / target_fps)]

    cap.release()
    return frames


def extract_clip_frames(total_frames, clip_range, frame_rate, target_fps=2, del_ratio=0.5):
    # 遍历clip_range，对每个切割片段做静态判断，如果是静态的，就直接删除这个片段
    target_fps = min(frame_rate, target_fps)
    target_clip_len = max(int(len(clip_range) * (1 - del_ratio)), 1)

    score_clip_dict = {}
    for range in clip_range[:]:
        # frames = []
        # frame_count = 0
        # for frame in total_frames[range[0]:range[1]]:
        #     if frame_count % int(frame_rate / target_fps) == 0:
        #         frames.append(frame)
        #     frame_count += 1
        range_frames = total_frames[range[0]:range[1]]
        frames = range_frames[::int(frame_rate / target_fps)]
        flow_maps = compute_farneback_optical_flow(frames)
        motion_score = downscale_and_average_flow_maps(flow_maps)

        motion_threshold = 1e-4

        # Filter out static scenes based on the threshold
        is_static_scene = motion_score < motion_threshold

        if is_static_scene:
            clip_range.remove(range)
            # print(f'remove {range}')
            # print("Motion Score:", motion_score)
            # print("Is Static Scene:", is_static_scene, '\n')
        else:
            score_clip_dict[motion_score] = range

    if len(clip_range) > target_clip_len:
        # 使用sorted()对字典的items进行排序，reverse=True表示降序
        sorted_dict = dict(sorted(score_clip_dict.items(),
                           key=lambda item: item[0], reverse=True))
        clip_range = list(sorted_dict.values())
        clip_range = clip_range[:target_clip_len]
        print(
            f"Remove some clips with lowest motion score, keep {target_clip_len} clips.\n")

    return clip_range

# 计算每两帧之间的光流图
def compute_farneback_optical_flow(frames):
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    flow_maps = []

    # Initialize Farneback optical flow parameters
    params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow_map = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, flow=None, **params)
        flow_maps.append(flow_map)
        prev_gray = gray
    return flow_maps


def downscale_and_average_flow_maps(flow_maps):
    downscaled_maps = [
        cv2.resize(
            flow, (16, int(flow.shape[0] * (16 / flow.shape[1]))), interpolation=cv2.INTER_AREA
        )
        for flow in flow_maps
    ]
    average_flow_map = np.mean(np.array(downscaled_maps), axis=0)
    return np.mean(average_flow_map)


if __name__ == '__main__':

    video_path = r"input_dir/4xbhusulyGM_scene-001.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path} is not a video file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if has_video_stream(video_path):
        frames = extract_frames(video_path)
        flow_maps = compute_farneback_optical_flow(frames)
        motion_score = downscale_and_average_flow_maps(flow_maps)

        motion_threshold = 0.1

        # Filter out static scenes based on the threshold
        is_static_scene = motion_score < motion_threshold

        print("Motion Score:", motion_score)
        print("Is Static Scene:", is_static_scene)
    else:
        print(f'{video_path} does not have a video stream')
