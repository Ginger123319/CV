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

