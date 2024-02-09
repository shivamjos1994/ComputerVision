import cv2
import mediapipe as mp
import time

# capturing the camera of the system
cap = cv2.VideoCapture(0)
pTime = 0


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec([0,255,0], thickness=1, circle_radius=1)


while cv2.waitKey(1) != 27:
    success, img = cap.read()
    
    if not success:
        break
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    # print(results)
    # if ihe image is showing landmarks then go through the loop.
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # drawing the landmarks and connect them on a face in the frame.
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
            # in mp.solutions.face_mesh module FACE_CONNECTIONS has been renamed to FACEMESH_TESSELATION in the latest version of mediapipe library.
            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                # x and y are the landmark's coordinates of a face.
                x, y = int(lm.x *iw), int(lm.y * ih)
                # print(id, x, y)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"fps: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("image", img)

cap.release()
