import cv2
import mediapipe as mp
import time


# mode = indicates whether to use static image mode or not.
# model_complexity = determines the complexity of the pose landmark model. Higher values are more accurate but slower. The possible values are 0, 1, or 2. The default value is 1.
# upBody = indicates whether to detect only the upper body or the full body. The default value is False, which means the full body is detected.
# smooth = indicates whether to use temporal smoothing or not. Temporal smoothing reduces jitter and improves stability of the pose landmarks. 
class poseDetector():
    def __init__(self, mode=False, model_complexity = 1, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils    
        self.mpPose = mp.solutions.pose    
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    # find the landmarks on the image
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
           # if the image has landmarks and draw is true then proceed, draw landmarks points and connect them on the image.
              self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    
    # find the particular landmark on the image.
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                # the coordinates of the landmark on the image.
                cx, cy =  int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                #  if draw is true then make a circle on the desired landmark that the user wants to see.
                   cv2.circle(img, (cx,cy), 2, (255, 0, 0), cv2.FILLED)
        return lmList







    
  


def main():
    cap = cv2.VideoCapture('PoseVideos/4.mp4')
    pTime = 0
    detector = poseDetector()

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




if __name__ == "__main__":
    main()