import cv2
import mediapipe as mp
import time
import math

class handDetector():
    # mode = a boolean value that indicates whether to use static image mode or not.
    # maxHands = an integer value that specifies the maximum number of hands to detect in the video frame.
    # model_complexity =  an integer value that determines the complexity of the hand landmark model. Higher values are more accurate but slower. The possible values are 0, 1, or 2. The default value is 1.
    # detectionCon = sets the minimum detection confidence threshold for the hand detection model. Values between 0 and 1 are accepted.
    # trackCon = sets the minimum tracking confidence threshold for the hand landmark model. Values between 0 and 1 are accepted.
    def __init__(self, mode=False, maxHands=2, model_complexity = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity 
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
        # mpDraw = provides some utility functions to draw the hand landmarks and bounding boxes on the video frames.
        self.mpDraw = mp.solutions.drawing_utils
        # the tip landmarks for the hand.
        self.tipIds = [4, 8, 12, 16, 20]


    # to show the hand's landmark with connected lines
    # draw = indicates whether to draw the hand landmarks and connections on the image or not. 
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        # if the list has landmarks, it'll show on the hand in the video.
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    

    # to find the position of the hand's landmark.
    # handNo = specifies which hand to process in the image. The default value is 0, which means the first detected hand.

    def findPosition(self, img, handNo=0, draw=True):
            # list contains the id and x, y position of each landmark on hand.
            self.lmList = []
            if self.results.multi_hand_landmarks:
                # myHand = assigns the hand landmarks object for the specified hand number from the self.results.multi_hand_landmarks list.
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    # calculate the x and y coordinates of the landmark on the image.
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])

                    if draw:
                    #  draw cicle on the landmarks on the image.
                       cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
            
            return  self.lmList
    

     
    # to find which finger is up or down.
    def fingersUp(self):
        fingers = []
        
        # Thumb is up can be said by having the tip to the left side of landmark just below the tip.(through x-axis, that's what [1] denotes)
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 fingers are up is said by having the tip 2 points below through y-axis.(that's what [2] denotes, the y-axis)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    

    # find the distance between the two points on the hand.
    def findDistance(self, p1, p2, img, draw=True, r=10, t=2):
        # x and y coordinates of the hand.
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        
        # center point between the two points.
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
            
            # length between the two points.
            length = math.hypot(x2-x1, y2-y1)

        return length, img, [x1, y1, x2, y2, cx, cy]









def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while cv2.waitKey(1) != 27:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        # if there's landmarks:
        if len(lmList) != 0:
            # show the landmark number 4, can change the landmark.
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 2,  (255,0,255), 2)

        cv2.imshow("image", img)  

    cap.release()




if __name__ == "__main__":
    main()