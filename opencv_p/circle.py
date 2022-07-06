import cv2
import numpy as np

path = r"..\..\source\opencv_pic\1.jpg"
path1 = r"..\..\source\opencv_pic\pre_mask.jpg"
img = cv2.imread(path)
img1 = cv2.imread(path1)
# print(img.shape, img1.shape)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
# print(mask.shape)
cut_pic = cv2.bitwise_and(img, img, mask=mask)
# print(cut_pic.shape)
cut_gray = cv2.cvtColor(cut_pic, cv2.COLOR_BGR2GRAY)
_, result_binary = cv2.threshold(cut_gray, 175, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
mask = cv2.morphologyEx(result_binary, cv2.MORPH_OPEN, kernel, iterations=1)
result = result_binary - mask
# print(result_binary.shape)
cv2.namedWindow("win", cv2.WINDOW_NORMAL)
cv2.imshow("win", result)
cv2.waitKey()
cv2.destroyAllWindows()
# img = cv2.imread(path, 0)
# img = cv2.medianBlur(img, 5)
# cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=5000, param1=150, param2=30, minRadius=800)
# print('circles1:', circles)
# circles = np.uint16(np.around(circles))
# print('circles2:', circles)
#
# for i in circles[0, :]:
#     # draw the outer circle
#     cv2.circle(cimg, (i[0], i[1]), i[2] - 50, (255, 255, 255), -1)
#     # draw the center of the circle
#     cv2.circle(cimg, (i[0], i[1]), 2, (255, 255, 255), 3)
# cv2.namedWindow("win", cv2.WINDOW_NORMAL)
# cv2.imshow("win", cimg)
# cv2.imwrite(r"..\..\source\opencv_pic\pre_mask.jpg", cimg)
# cv2.waitKey()
# cv2.destroyAllWindows()
