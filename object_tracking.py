import cv2
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from base64 import b64encode
import urllib
import zipfile
import requests

video_input_file_name = "528_528-0360_preview.mp4"


def drawRectangle(frame, bbox):
   p1 = (int(bbox[0]), int(bbox[1]))
   p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
   cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

def displayRectangle(frame, bbox):
   plt.figure(figsize=(20,10))
   frameCopy = frame.copy()
   drawRectangle(frameCopy, bbox)
   frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
   plt.imshow(frameCopy); plt.axis('off')

def drawText(frame, txt, location, color = (50, 170, 50)):
   cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)



# if not os.path.isfile('goturn.prototxt') or not os.path.isfile('goturn.caffemodel'):
#     print("Downloading GOTURN model zip file")
#     # urllib.request.urlretrieve('https://www.dropbox.com/sh/77frbrkmf9ojfm6/AACgY7-wSfj-LIyYcOgUSZ0Ua?dl=1', 'GOTURN.zip')
    
#     zip_file_url = 'https://www.dropbox.com/sh/77frbrkmf9ojfm6/AACgY7-wSfj-LIyYcOgUSZ0Ua?dl=1'
#     zip_file_path = 'GOTURN.zip'
#     extract_path = '.'

#     response = requests.get(zip_file_url)
#     with open(zip_file_path, 'wb') as zip_file:
#          zip_file.write(response.content)

    
#     # Extract the contents of the ZIP file
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_path)

 
#     print(f"Contents extracted to: {extract_path}")

#     # Delete the ZIP file
#     os.remove(zip_file_path)
#     print(f"ZIP file '{zip_file_path}' deleted.")

    # # Uncompress the file
    # !tar -xvf GOTURN.zip

    # # Delete the zip file
    # os.remove('GOTURN.zip')



# creating the tracker instance:
   
# Set up tracker
tracker_types = ['BOOSTING', 'MIL','KCF', 'CSRT', 'TLD', 'MEDIANFLOW', 'GOTURN','MOSSE']

# Change the index to change the tracker type
tracker_type = tracker_types[1]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy_TrackerBoosting.create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy_TrackerCSRT.create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy_TrackerTLD.create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy_TrackerMedianFlow.create()
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
else:
    tracker = cv2.legacy_TrackerMOSSE.create()





# read input video and setup output video:
# Read video
video = cv2.VideoCapture(video_input_file_name)
ok, frame = video.read()

# Exit if video not opened
if not video.isOpened():
    print("Could not open video")
    sys.exit()
else :
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_output_file_name = 'race_car-' + tracker_type + '.mp4'
video_out = cv2.VideoWriter(video_output_file_name,cv2.VideoWriter_fourcc(*'avc1'), 10, (width, height))



# Define a bounding box
bbox = (170, 220, 860, 200)
#bbox = cv2.selectROI(frame, False)
#print(bbox)
displayRectangle(frame,bbox)



# Initialize tracker:
# initialize tracker with one frame and bounding box.
ok = tracker.init(frame, bbox)




# read frame and track object:
while True:
  ok, frame = video.read()
  if not ok:
    break

  # start timer:
  timer = cv2.getTickCount()

  # update tracker:
  ok, bbox = tracker.update(frame)

  # calculate frame per second(FPS):
  fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

  # draw bounding box:
  if ok:
    drawRectangle(frame, bbox)
  else:
    drawText(frame, "tracking failure detected", (80, 140), (0,0,255))

  # display info:
  drawText(frame, tracker_type + "Tracker", (80,60))
  drawText(frame, "fps: " + str(int(fps)), (80,100))

  # write frame to video:
  video_out.write(frame)

  # display the result
  cv2.imshow("Tracking", frame)

  # Exit if 'ESC' key is pressed
  if cv2.waitKey(1) & 0xFF == 27:
      break


video.release()
video_out.release()
cv2.destroyAllWindows()

