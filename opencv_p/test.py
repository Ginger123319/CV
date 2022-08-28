<<<<<<< HEAD
import cv2

path = r"D:\Python\source\arm_data\outputs\attachments\1636527059_757567_1.png"
path1 = r"..\..\source\opencv_pic\pre_mask.jpg"
src = cv2.imread(path)
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, src = cv2.threshold(src, 50, 255, cv2.THRESH_BINARY)
# dst = cv2.GaussianBlur(src, (5, 5), 0)
# dst = cv2.addWeighted(src, 2, dst, -1, 0)
cv2.namedWindow("src show", cv2.WINDOW_NORMAL)
cv2.imshow("src show", src)
# cv2.namedWindow("dst show", cv2.WINDOW_NORMAL)
# cv2.imshow("dst show", dst)
cv2.waitKey(0)
=======
# import cv2

# path = r"..\..\source\opencv_pic\1.png"
# path1 = r"..\..\source\opencv_pic\pre_mask.jpg"
# src = cv2.imread(path)
# # src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# dst = cv2.GaussianBlur(src, (5, 5), 0)
# dst = cv2.addWeighted(src, 2, dst, -1, 0)
# cv2.namedWindow("src show", cv2.WINDOW_NORMAL)
# cv2.imshow("src show", src)
# cv2.namedWindow("dst show", cv2.WINDOW_NORMAL)
# cv2.imshow("dst show", dst)
# cv2.waitKey(0)
>>>>>>> 8e573c5ca72d22baed81e334b509533fd6d7a85a

# a = 45 ^ 255
# print(a)
# print(False in (src == dst))


<<<<<<< HEAD
# from PIL import Image, ImageDraw, ImageFont
#
# image = Image.new(mode='RGBA', size=(400, 50))
# draw_table = ImageDraw.Draw(im=image)
# # draw_table.text(xy=(0, 0), text=u'仰起脸笑得像满月', fill='#008B8B', font=ImageFont.truetype('./SimHei.ttf', 50))
#
# image.show()  # 直接显示图片
# image.save('满月.png', 'PNG')  # 保存在当前路径下，格式为PNG
# image.close()
=======
from PIL import Image, ImageDraw, ImageFont

image = Image.new(mode='RGBA', size=(400, 50))
draw_table = ImageDraw.Draw(im=image)
# draw_table.text(xy=(0, 0), text=u'仰起脸笑得像满月', fill='#008B8B', font=ImageFont.truetype('./SimHei.ttf', 50))

image.show()  # 直接显示图片
image.save('满月.png', 'PNG')  # 保存在当前路径下，格式为PNG
image.close()
>>>>>>> 8e573c5ca72d22baed81e334b509533fd6d7a85a
