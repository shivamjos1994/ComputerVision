import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("AiTrainer/curls.mp4")
detector = pm.poseDetector()
pTIme = 0
count = 0
# direction is up initially.
dir = 0

while cv2.waitKey(1) != 27:
    success, img = cap.read()
    # img = cv2.imread("AiTrainer/test.jpg")
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    # if the frame has landmarks, then proceed.
    if len(lmList) != 0:
        # for the right arm, 12, 14, 16 are the three landmarks points on shoulder, elbow and wrist respectively.
        # detector.findAngle(img, 12, 14, 16)
        # for the left arm.
        angle = detector.findAngle(img, 11, 13, 15)

        # converting the minimum(210) and maximum(310) angle to 0 and 100 percentage respectively.
        per = np.interp(angle, (210, 310), (0, 100))
        # print(angle, per)
        
        # converting the min and max angles to 500 (min value = 650) and 100 respectively.
        bar = np.interp(angle, (210, 310), (500, 100))


        # check for the dumbbell curls:
        # here complete movement from 0-100 and 100-0 count as one rep of the exercise.
        # (dir = 0) means hand is up, (dir = 1) means hand is down.
        color = (255, 0, 0)
        if per == 100:
            color = (0, 255, 0)
            # hand is up
            if dir == 0:
                # half count
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 0, 255)
            # hand is down.
            if dir == 1:
                # count is half. Now it becomes one count.
                count += 0.5
                dir = 0
        # print(count)
        
        # Draw bar.
        cv2.rectangle(img, (900, 100), (925, 500), color, 2)
        cv2.rectangle(img, (900, int(bar)), (925, 500), color, cv2.FILLED )
        cv2.putText(img, f'{int(per)}%', (870, 85), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)


        # Draw curl count.
        cv2.rectangle(img, (30, 400), (160, 500), (0, 255, 0), cv2.FILLED )
        cv2.putText(img, f'{int(count)}', (70, 460), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 4)
    

    cTIme = time.time()
    fps = 1 /(cTIme - pTIme) 
    pTIme = cTIme
    

    cv2.putText(img, f'FPS: {int(fps)}', (30,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 4)
    cv2.imshow("image", img)

cap.release()