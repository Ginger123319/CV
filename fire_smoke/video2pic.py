import ffmpeg
import shutil
import os
import math

video_path = '/home/room/视频/炼化/2020-9-3验收报警/fire/炼化厂区一部/P1004-AB泵-常压-一部/1599133275'
for video_name in os.listdir(video_path):
    if video_name.endswith('.mp4'):
        print(video_name)
        YOUR_FILE = os.path.join(video_path, video_name)
        PICPERMIN = 60
        # ----------------------------------------------------
        # 关键帧提取
        if not os.path.exists("/home/room/TEST_JYF/key_frame_fire"):
            os.mkdir("/home/room/TEST_JYF/key_frame_fire")
        probe = ffmpeg.probe(YOUR_FILE)
        width = 480  # default 480

        info = None
        for i in probe["streams"]:
            if "width" in i.keys():
                info = i
                break
        # Set how many spots you want to extract a video from.
        if info is None:
            print("cannot find a video stream")
            exit(-1)

        width = info["width"]
        time = math.ceil(float(info['duration'])) - 1

        parts = math.ceil(time / (60 / PICPERMIN))
        intervals = time // parts
        intervals = math.ceil(intervals)
        interval_list = [(i * intervals, (i + 1) * intervals) for i in range(parts)]
        i = 0

        for item in interval_list:
            (
                ffmpeg
                .input(YOUR_FILE, ss=item[1])
                .filter('scale', width, -1)
                .output('/home/room/TEST_JYF/key_frame_fire/' + video_name + '_' + str(i) + '.jpg', vframes=1)
                .run()
            )
            i += 1

