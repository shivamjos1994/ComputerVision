import cv2
import mediapipe as mp
import time
import FaceDetectionModule as fdm



cap = cv2.VideoCapture('Videos/2.mp4')
pTime = 0
detector =fdm.FaceDetector()

while cv2.waitKey(15) != 27:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
#  print(bboxs)

        
    cTime = time.time()
    try:     
        fps = 1 / (cTime - pTime)
        pTime = cTime
    except ZeroDivisionError:
        fps = 0

    cv2.putText(img, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow('Image', img)

cap.release()
