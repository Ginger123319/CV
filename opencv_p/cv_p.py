import cv2

# 读取
img = cv2.imread("sources/59.jpg")
cv2.imshow("img_1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 绘图
cv2.line(img, (100, 100), (200, 100), color=(0, 0, 255), thickness=3)
cv2.imshow("line", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图片通道变换(通过阈值改变像素点的值)-RGB转灰度再转为二值图（非黑即白，一个通道）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 自适应阈值，局部像素点使用的局部的阈值，而不是只使用一个全局阈值
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 0)
cv2.imshow("binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
