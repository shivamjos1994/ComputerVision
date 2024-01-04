import cv2
import sys

# default index camera for the video to capture.
s = 0

# user can specify the index of the camera or name of the file as an argument when running the script.
if len(sys.argv) > 1:
  s = sys.argv[1]

# VideoCapture class implments the video capture functionality and can read frames from a camera or a file.
source = cv2.VideoCapture(s)

# name of the window that'll display the video.
win_name = 'Camera Preview'

# the cv2.namedWindow function creates a window with the given name and flags.
# cv2.WINDOW_NORMAL is a flag which indicates that the window can be resized by the user.
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)


# ASCII code for esc key is 27. 
while cv2.waitKey(1) != 27:
    #  The has_frame variable stores the boolean value and the frame variable stores the array.
     has_frame, frame = source.read()
     if not has_frame:
       break
    # The frame argument is the image to be displayed, which is set to the frame variable.
     cv2.imshow(win_name, frame)
    
source.release()

#The cv2.destroyWindow function destroys the specified window and frees the memory associated with it.
cv2.destroyWindow(win_name)