import cv2
import numpy as np
import time
import pyautogui
import handTrackingModule as htm

###############################

wCam, hCam = 640, 480
frameR = 100  # frame reduction
smoothening = 5

###############################

pTime = 0

# previous and current location of x and y.
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
# width and height of the screen.
wSCr, hScr = pyautogui.size()
# print(wSCr, hScr)

while cv2.waitKey(1) != 27:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    img = detector.findHands(img)

    # landmarks of the detected hands.
    lmList = detector.findPosition(img, draw=False)

    # get tip of the index and middle finger.
    if len(lmList) != 0:
        # x and y coordinates of the index and middle finger.
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1,y1,x2,y2)

        # check which fingers are up.
        fingers = detector.fingersUp()
        # print(fingers)
        
        # only index finger: moving mode
        if fingers[1]==1 and fingers[2]==0:

            # convert coordinates.(width and height of the frame to be converted to the width and height of the frame.)
            x3 = np.interp(x1, (0,wCam), (0,wSCr))
            y3 = np.interp(y1, (0,wCam), (0,wSCr))

            # smoothening the values, in order to run the cursor smoother.
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening

            # move mouse
            pyautogui.moveTo(cLocX, cLocY)
            cv2.circle(img, (x1,y1), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1,y1), 18, (255, 0, 255), 2)

            # updating the location.
            pLocX, pLocY = cLocX, cLocY

        # both index and middle fingers are up: selecting mode.
        if fingers[1]==1 and fingers[2]==1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # lineInfo contains the coordinates of both the fingers and the center coordinates also.
            # print(length)
            # if the distance between the index and middle finger is short, then click the mouse.
            if length < 40:
                cv2.circle(img, (x1,y1), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2,y2), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
                time.sleep(1)

    

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.imshow("Image", img)

cap.release()
