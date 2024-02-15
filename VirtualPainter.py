import cv2
import numpy as np
import time
import os
import handTrackingModule as htm

################################

brushThickness = 15
eraserThickness = 100

################################




folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

# appending the images in the overlaylist.
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]
# by default the draw color is orange.
drawColor = (0, 165, 255)


cap = cv2.VideoCapture(0)
# setting the width and height.
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

# initial points for the index finger are 0.
xp, yp = 0, 0

# creating a canvas image to draw on, because the frame keeps updating, so we can't draw on the frame.
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while cv2.waitKey(1) != 27:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    # if hands got detected then proceed further.
    if len(lmList) != 0:
        # tip coordinates of index finger.
        x1, y1 = lmList[8][1:]
        # tip coordinates of the middle finger.
        x2, y2 = lmList[12][1:]

        # checking if the finger is up.
        fingers = detector.fingersUp()
        # print(fingers)

        # Selectin mode - if two fingers are up.(index and middle finger for the right hand.)
        if fingers[1] and fingers[2]:
            # whenever there's selection, make the coordinates of the index tip 0.
            xp, yp = 0, 0
            # print("Selection mode")
            # if we're in the header, then proceed further.(checking for the click)
            if y1 < 122:
                # it'll overlay the images as per the condition.
                if 250 < x1 <450:
                    header = overlayList[0]
                    drawColor = (0, 165, 255)
                elif 550 < x1 <750:
                    header = overlayList[1]
                    drawColor = (255, 255, 255)
                elif 800 < x1 <950: 
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 <1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1-30), (x2, y2+30), drawColor, cv2.FILLED)            

        
        # Drawing mode - if only the index finger is up.
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            # print("Drawing mode")
            # if the initial points on the very first frame are 0 then update it to wherever the coordinates of index finger are.
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # to have some eraser thickness.
            if drawColor == (0, 0, 0):
               cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
               cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
               cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
               cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            # update the points.
            xp, yp = x1, y1

    
    # merging the canvas image to the frame in order to draw on the frame.
    # Step1: converting the canvas image to the gray channel
    # Step2: create an inverse (binary image) out of gray scale canvas image.
    # Step3: convert the inverse image to the BGR format.
    # Step4: combine both the inverse and the original frame using bitwise_and operator.
    # Step5: combine the resultant image with the canvas image using bitwise_or operator.  
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # overlay the header image on the frame. (our header is of size 1280 X 122)
    img[0:122, 0:1280] = header


    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)

cap.release()