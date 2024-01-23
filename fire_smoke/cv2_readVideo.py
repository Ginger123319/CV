import cv2

cap = cv2.VideoCapture('/home/room/TEST_JYF/test_precision1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('image', frame)
    print(frame.shape)
    # HWC-(1080,1920,3)
    k = cv2.waitKey(20)
    # q键退出
    if k & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
