import cv2
import mediapipe as mp
import time


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('PoseVideos/4.mp4')
pTime = 0


while cv2.waitKey(1) != 27:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            # print(id, lm)
            cx, cy =  int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx,cy), 2, (255, 0, 0), cv2.FILLED)


    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 2,  (255,0,255), 2)

    cv2.imshow("image", img)    

cap.release()
