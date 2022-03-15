import cv2

# opencv基本使用与pil的使用方法基本类似
img = cv2.imread("img01.jpg")
# cv2.imshow("img1", img)
# cv2.waitKey(0)
# cv2.destroyWindow("img1")

color = (255, 0, 0)
cv2.line(img, (20, 20), (100, 100), color, 3)
cv2.rectangle(img, (20, 20), (100, 100), color, 3)
cv2.imshow("img1", img)
cv2.waitKey(0)
# cv2.destroyWindow()

