import os
import cv2
import numpy as np

path = r"G:\liewei\source\2actions"
for folder in os.listdir(path):
    for video in os.listdir(os.path.join(path, folder)):
        video_name = os.path.join(path, folder, video)
        print(video_name)
        vc = cv2.VideoCapture(video_name)
        count = 0
        save_count = 0
        while vc.isOpened():
            # get a frame
            ret, frame = vc.read()
            if frame is None:
                break
            # show a frame
            if ret:
                count += 1
                if count % 4 == 0:
                    save_count += 1
                    frame = cv2.resize(frame, (240, 320))
                    # frame = detect(frame)
                    cv2.imshow("video", frame)
                    if cv2.waitKey(40) & 0xFF == ord('q'):
                        break

        vc.release()
        cv2.destroyAllWindows()
        print(count, save_count)
        exit()
