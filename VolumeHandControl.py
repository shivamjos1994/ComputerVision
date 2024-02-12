# install the pycaw module, if not installed already, helps in controlling of the volume of the system.
import cv2
import time
import numpy as np
import handTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume




# width and height for the webcam frame.
wCam , hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)


##########################################
# pycaw code for volume control.

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()

# maximum and minimum volume.
minVol = volRange[0]    # -65
maxVol = volRange[1]    # 0 


##########################################

vol = 0
volBar = 400
volPer = 0

while cv2.waitKey(1) != 27:
    success, img = cap.read()
    
    if not success:
        break
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # tip of thumb and index finger.
        # print(lmList[4], lmList[8])

        # x and y coordinates of tip of thumb and index finger. 
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # center between two points.
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # highlighting the desired landmark points.
        cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # line joining the two points.
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # length between the two points.
        length = math.hypot(x2-x1, y2-y1)

        # Conversion of hand range to volume range. 
        # Hand Range 30 - 220
        # Volume Range -65 - 0

        # converting the hand range with volume range.
        # the volume bar of the system.
        vol = np.interp(length, [30, 220], [minVol, maxVol])
        # volume bar on the frame.
        volBar = np.interp(length, [30, 220], [400, 150])
        # volume percentage on the frame.
        volPer = np.interp(length, [30, 220], [0, 100])

        print(int(length), vol)    # printing the length and volume.

        # setting the master volume level.
        volume.SetMasterVolumeLevel(vol, None)

        # if length between two landmarks is less or greater than certain value, change the color accordingly.
        if length <= 30:
           cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
        elif length >= 220:
           cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
    
    # the volume bar on the frame.
    cv2.rectangle(img, (40, 150), (50, 400), (255, 0, 0), 3)   # empty volume bar.
    cv2.rectangle(img, (40, int(volBar)), (50, 400), (255, 0, 0), cv2.FILLED)    # filled volume bar according to the increasing and decreasing volume.
    cv2.putText(img, f"{int(volPer)} %", (35, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)   # volume percentage.

           
           
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS:{int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2 )
    cv2.imshow("Image", img)

cap.release()