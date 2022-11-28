import cv2

# 导入级联分类器文件
faceCascade = cv2.CascadeClassifier(r'C:\Users\Monty _L\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')

imgGray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
imgGray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 检测人脸，返回人脸矩阵四个角的坐标
face1 = faceCascade.detectMultiScale(imgGray1, 1.1, 4)
face2 = faceCascade.detectMultiScale(imgGray2, 1.2, 1)

# 绘制矩形框
for (x, y, w, h) in face1:
    cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)
for (x, y, w, h) in face2:
    cv2.rectangle(img2, (x, y), (x + w, y + h), (245, 148, 15), 2)

cv2.imshow("Result1", img1)
cv2.imshow("Result2", img2)
cv2.waitKey(0)
