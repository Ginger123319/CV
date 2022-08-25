import cv2
import matplotlib.pyplot as plt

file_path = r"D:\Python\source\arm_data\1636527059_757567.jpg"
img = cv2.imread(file_path)
cut_pic = cv2.medianBlur(img, 5)
img_gray = cv2.cvtColor(cut_pic, cv2.COLOR_BGR2GRAY)
ret, thresh2 = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
# 闭操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
close_dst = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=1)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
thresh2 = cv2.erode(close_dst, kernel)
thresh2 = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)
cut_pic = cv2.bitwise_and(img, thresh2)
cut_pic = cv2.medianBlur(cut_pic, 5)
cut_gray = cv2.cvtColor(cut_pic, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(cut_gray, (3, 3), 0)
canny = cv2.Canny(img, 20, 150)
cv2.namedWindow("img_contour", cv2.WINDOW_NORMAL)
cv2.imshow("img_contour", cut_pic)
cv2.waitKey(0)
# ret, thresh3 = cv2.threshold(cut_gray, 140, 255, cv2.THRESH_BINARY)
# # 开操作
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# close_dst = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel, iterations=1)
# # contours, hierarchy = cv2.findContours(thresh3, cv2.RETR_TREE,
# #                                        cv2.CHAIN_APPROX_SIMPLE)
# # img_contour = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# cv2.namedWindow("img_contour", cv2.WINDOW_NORMAL)
# cv2.imshow("img_contour", cut_pic)
# cv2.waitKey(0)
