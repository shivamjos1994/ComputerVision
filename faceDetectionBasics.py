import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture('Videos/1.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)



while cv2.waitKey(15) != 27:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # by default we can detect faces with mpDraw.
            # mpDraw.draw_detection(img, detection)

            # print(id, detection)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box   # bounding box coming from class.
            ih, iw, ic = img.shape
            # the bounding box coordinates.
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            # confidence score 
            cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    
    cTime = time.time()
    try: 
       fps = 1 / (cTime - pTime)
       pTime = cTime
    except ZeroDivisionError:
        fps = 0

    cv2.putText(img, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow('Image', img)

cap.release()