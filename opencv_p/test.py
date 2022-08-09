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

# a = 45 ^ 255
# print(a)
# print(False in (src == dst))


from PIL import Image, ImageDraw, ImageFont

image = Image.new(mode='RGBA', size=(400, 50))
draw_table = ImageDraw.Draw(im=image)
# draw_table.text(xy=(0, 0), text=u'仰起脸笑得像满月', fill='#008B8B', font=ImageFont.truetype('./SimHei.ttf', 50))

image.show()  # 直接显示图片
image.save('满月.png', 'PNG')  # 保存在当前路径下，格式为PNG
image.close()
