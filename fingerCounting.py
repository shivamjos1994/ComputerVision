import cv2
import time
import os
import handTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


folderPath = "FingerImages"
myList = os.listdir(folderPath)
# print(myList)

# create a list and append the images into it.
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

pTime = 0

detector = htm.handDetector(detectionCon=0.75)

# The landmarks on the hand.
# 4 = tip of the thumb.
# 8 = tip of the index finger.
# 12 = tip of the middle finger.
# 16 = tip of the ring finger.
# 20 = tip of the pinkie finger.
tipIds = [4, 8, 12, 16, 20]

while cv2.waitKey(1) != 27:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
       fingers = []
       # for thumb.()
        # '<' is actually for the left hand, but we've flipped the frame so it'll work on the right hand as well. Change comparison operator..
        #   ... to see the effects on both hands. (it'll work properly for one hand only.)
    #  here [1] means that it'll compare the landmarks from the x-axis, weather to be on left or right.
       if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
           fingers.append(1)
       else:
           fingers.append(0)    
       # for 4 fingers.    
       for id in range(1, 5):
        # if the tip of the finger is 2 below the landmark. (that's what [2] represents, the y-axis.) 
           if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
               fingers.append(1)
           else:
               fingers.append(0)
    #    print(fingers)
       totalFingers = fingers.count(1)
    #    print(totalFingers)
       
    # height, width and channels of the image to match all the images and place the image on the top left corner of he frame.
    # (totalFingers-1) represents that if the user is showing 2 fingers, then the image shown will be on the (2-1) index.
       h, w, c = overlayList[totalFingers-1].shape
       img[0:h, 0:w] = cv2.resize(overlayList[totalFingers-1], (w, h))
       
    # the rectangle and the count of fingers inside the rectangle.
       cv2.rectangle(img, (10, 250), (140, 450), (0, 255, 0), cv2.FILLED)
       cv2.putText(img, str(totalFingers), (45, 360), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 4)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS:{int(fps)}', (480, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("img", img)

cap.release()
