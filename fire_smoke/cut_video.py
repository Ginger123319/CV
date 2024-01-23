import os
import cv2


def single(single_path):
    # 单个视频处理
    cap = cv2.VideoCapture(single_path)
    cap.isOpened()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(width, height)

    if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
        # get方法参数按顺序对应下表（从0开始编号)
        rate = cap.get(5)  # 帧速率
        print(rate)

        FrameNumber = int(cap.get(7))  # 视频文件的帧数
        print(FrameNumber)

        duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
        print(duration)

        fps = int(FrameNumber / 2 + 1)  # 每一段小视频帧数,此处希望三等分
        print(fps)

        i = 0
        while True:
            success, frame = cap.read()
            if success:
                i += 1
                if i % fps == 1:
                    print(i)
                    videoWriter = cv2.VideoWriter(
                        '/home/room/TEST_JYF/测试数据/gas/' + 'kun' + '_' + str(i) + '.mp4',
                        cv2.VideoWriter_fourcc(*'mp4v'), rate,
                        (int(width), int(height)))
                    videoWriter.write(frame)
                else:
                    if i % 50 == 1:
                        # print("%10:{}".format(i))
                        videoWriter.write(frame)
                    else:
                        continue
            else:
                print('end')
                break

    cap.release()


def multi(mul_path_list):
    # 批量切割视频
    for epoch in range(len(mul_path_list)):
        for video_name in os.listdir(mul_path_list[epoch]):
            print(video_name)

            cap = cv2.VideoCapture(os.path.join(mul_path_list[epoch], video_name))
            cap.isOpened()
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            print(width, height)

            if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
                # get方法参数按顺序对应下表（从0开始编号)
                rate = cap.get(5)  # 帧速率
                print(rate)

                FrameNumber = int(cap.get(7))  # 视频文件的帧数
                print(FrameNumber)

                duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
                print(duration)

                fps = int(FrameNumber / 2 + 1)  # 每一段小视频帧数
                print(fps)

                i = 0
                while True:
                    success, frame = cap.read()
                    if success:
                        i += 1
                        if i % fps == 1:
                            print(i)
                            videoWriter = cv2.VideoWriter(
                                f'/home/room/TEST_JYF/cut/gas{epoch}/' + video_name.split('.')[0] + '_' + str(
                                    i) + '.mp4',
                                cv2.VideoWriter_fourcc(*'mp4v'), rate,
                                (int(width), int(height)))
                            videoWriter.write(frame)
                        else:
                            if i % 50 == 1:
                                # print("%10:{}".format(i))
                                videoWriter.write(frame)
                            else:
                                continue
                    else:
                        print('end')
                        break
            cap.release()


if __name__ == '__main__':
    path = '/home/room/gas-station-video-2022.9.14/0330-昆山14-12号14路视频/4_0.mp4'
    path2 = '/home/room/gas-station-video-2022.9.14/0329-加油区收现金视频/昆山'
    path3 = '/home/room/gas-station-video-2022.9.14/0331昆山9-11号14路加油区视频'
    path4 = '/home/room/100路摄像头视频/153/20210205/01'
    path_list = [path2, path3]
    multi(path_list)
