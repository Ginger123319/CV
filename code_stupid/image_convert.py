from PIL import Image
import cv2
import numpy as np

# PIL有九种不同模式: 1，L，P，RGB，RGBA，CMYK，YCbCr，I，F
# 参考：https://www.jianshu.com/p/0644aad17287
img = Image.open(r'D:\Python\test_jzyj\key_mouse_20\pic\0005.jpg').convert('F')
img = img.convert('RGB')
img.show()

# 还是RGB，想要和cv2读出来的值一模一样，还是要进行颜色通道的转换
img = np.array(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
print(img.shape)
exit()
cv2.imshow("BGR", img)
cv2.waitKey()

img = cv2.imread(r'D:\Python\test_jzyj\key_mouse_20\pic\0005.jpg')
print(img)
cv2.imshow("BGR", img)
cv2.waitKey()
