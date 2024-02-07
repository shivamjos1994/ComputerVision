import cv2
import mediapipe as mp
import time
# importing the customized module 
import handTrackingModule as htm

# previous time
pTime = 0
# current time
cTime = 0

cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while cv2.waitKey(1) != 27:
    success, img = cap.read()
    # no landmarks points will be shown to you.
    # img = detector.findHands(img, draw=False)
    img = detector.findHands(img)
    
    # no landmark's position will be shown to you.
    # lmList = detector.findPosition(img, draw=False)
    lmList = detector.findPosition(img)
    
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 2,  (255,0,255), 2)

    cv2.imshow("image", img)  

cap.release()   