# 矫正人脸框：原始框不太标准（框偏大），适当缩放方法------------★★★★★

from PIL import Image
from PIL import ImageDraw
import os

# 图片和标签路径
IMG_DIR = r".\img_celeba"
AND_DIR = r".\Anno"

# 人脸框的矫正系数
a1 = 0.12
a2 = 0.1
a3 = 0.9
a4 = 0.85

# 人脸框的坐标和宽高（校正前）：共测试了5张图片
x1_ = [95,72,216,622,236]
y1_ = [71,94,59,257,109]
w_ = [226,221,91,564,120]
h_ = [313,306,126,781,166]

# 图片名称
names = ["000001.jpg","000002.jpg","000003.jpg","000004.jpg","000005.jpg"]

# 人脸框矫正
for i in range(len(x1_)):
    x1 = x1_[i]
    y1 = y1_[i]
    w = w_[i]
    h = h_[i]
    name = names[i]

    img = Image.open(os.path.join(IMG_DIR,name))
    imgdraw = ImageDraw.Draw(img)
    imgdraw.rectangle((x1,y1,x1+w,y1+h),outline="blue",width=3)                # 矫正前
    imgdraw.rectangle((x1+w*a1,y1+h*a2,x1+w*a3,y1+h*a4),outline="red",width=3) # 矫正后

    img.show()
    # exit()