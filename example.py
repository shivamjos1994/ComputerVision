import cv2
import mediapipe as mp
import time
import FaceMeshModule as fmm


cap = cv2.VideoCapture(0)
pTime = 0
detector = fmm.FaceMeshDetector()

while cv2.waitKey(1) != 27:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    img, faces = detector.findFaceMesh(img)
    if len(faces) != 0:
        print(len(faces))


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"fps: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("image", img)

cap.release()


