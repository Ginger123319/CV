
'''
PIL在图片上写中文
'''
# # coding:utf-8
# from PIL import Image, ImageDraw, ImageFont
#
# image= Image.new('RGB', (559, 320),(255,255,255))
# draw = ImageDraw.Draw(image)
#
# # draw.text()
# font = ImageFont.truetype("arial", 40, encoding="unic") # 设置字体
# draw.text((100, 50), "哈哈哈", 'black', font)
# # del draw
# image.show()
# # printers = win32print.EnumPrinters(10)
# # print printers
# coding:utf-8

from PIL import Image, ImageDraw, ImageFont

image= Image.new('RGB', (559, 320),(255,255,255))
draw = ImageDraw.Draw(image)

# draw.text()
font = ImageFont.truetype("simsun.ttc", 40, encoding="unic") # 设置字体
draw.text((100, 50), "哈哈哈", 'black', font)
# del draw
image.show()
# printers = win32print.EnumPrinters(10)
# print printers