import cv2
import mediapipe as mp
import time
import PoseModule as pm

cap = cv2.VideoCapture('PoseVideos/4.mp4')
pTime = 0
detector = pm.poseDetector()

while cv2.waitKey(1) != 27:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
    # finding the right elbow, because 14 is right elbow landmark.
        if len(lmList) != 0:
            # print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 5, (0, 255, 0), cv2.FILLED)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 2,  (255,0,255), 2)

        cv2.imshow("image", img)    
 
cap.release()
