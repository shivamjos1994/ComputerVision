import cv2
import mediapipe as mp
import time

# this module can be imported and used for face mesh for any file.
# staticMode = indicates whether to use static image mode or not.
# maxFaces = the number of faces it can detect.
class FaceMeshDetector():
    def  __init__(self, staticMode=False, maxFaces=2):
         self.staticMode =  staticMode
         self.maxFaces = maxFaces

         self.mpDraw = mp.solutions.drawing_utils
         self.mpFaceMesh = mp.solutions.face_mesh
         self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces)
        #  color, thickness and circle radius for the landmarks on the face can be drawn from here.
         self.drawSpec = self.mpDraw.DrawingSpec([0,255,0], thickness=1, circle_radius=int(0.5))

    
    def findFaceMesh(self, img, draw=True):
         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         self.results = self.faceMesh.process(imgRGB)

         faces = []
         # if ihe image is showing landmarks then go through the loop.
         if self.results.multi_face_landmarks:
             for faceLms in self.results.multi_face_landmarks:
                #  if the face is detecting and has landmarks, also draw=True, then draw the landmarks and connect them on a face in the frame.
                 if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                 # in mp.solutions.face_mesh module FACE_CONNECTIONS has been renamed to FACEMESH_TESSELATION in the latest version of mediapipe library.
                 face = []
                 for id, lm in enumerate(faceLms.landmark):
                     ih, iw, ic = img.shape
                     x, y = int(lm.x *iw), int(lm.y * ih)
                    #  face is a list for one face only.
                     face.append([x, y])
                # faces is a list for multiple faces.
                 faces.append(face)
         return img, faces
                    
        

    

    



def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()

    while cv2.waitKey(1) != 27:
        success, img = cap.read()

        if not success:
            break
        img = cv2.flip(img, 1)

        img, faces = detector.findFaceMesh(img)
        # if there's atleast one face, then printing the number of faces. (max number of faces it can detect is 2.)
        if len(faces) != 0:
            print(len(faces))


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"fps: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("image", img)

    cap.release()






if __name__ == "__main__":
    main()