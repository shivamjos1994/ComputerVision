import os
import cv2 
import numpy as np
import urllib.request
import tarfile
import matplotlib.pyplot as plt


# creating config file from the frozen graph.
modelFile = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
configFile = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = "coco_class_labels.txt"
model_dir = 'models'

# if models directory not present, make one.
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

# if paticular file not present download it.
if not os.path.isfile(modelFile):
    os.chdir(model_dir)
    # Download the tensorflow Model
    urllib.request.urlretrieve('http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz', 'ssd_mobilenet_v2_coco_2018_03_29.tar.gz')
    
    # Uncompress the file
    with tarfile.open('ssd_mobilenet_v2_coco_2018_03_29.tar.gz', 'r:gz') as tar:
         tar.extractall()

    # Delete the tar.gz file
    os.remove('ssd_mobilenet_v2_coco_2018_03_29.tar.gz')

    # Come back to the previous directory
    os.chdir("..")


# check class labels:
with open(classFile) as fp:
    labels = fp.read().split("\n")
# print(labels)
    

""" Steps for performing inference using DNN model:
1. load the model and input image into memory
2. detect objects using a forward pass through a network.
3.display the detected objects with bounding boxes and class labels."""

# read the tensorflow model:
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)


# detect objects in an image using a pre-trained neural network
def detect_objects(net, im):
    #  represents the target size of the image that will be fed to the network. The image will be resized to a square of 200 by 200 pixels.
    dim = 200

    # create a blob from the image:
    # im: the input image that will be converted to a blob.
    # 1.0: the scale factor that multiplies the image pixel values. It is used to normalize the pixel values to a certain range, such as 0-1 or 0-255.
    # size=(dim, dim): the target size that the image will be resized to. It is a tuple of two integers that represent the width and height of the image in pixels. In this case, it is the same as the value of dim.
    # mean=(0,0,0): the mean values that are subtracted from each channel of the image. It is a tuple of three values that represent the mean of the blue, green, and red channels. In this case, it is zero, which means no mean subtraction is performed.
    # swapRB=True: a boolean flag that indicates whether to swap the red and blue channels of the image.
    # crop=False: a boolean flag that indicates whether to crop the image from the center. If True, the image is cropped to the target size without resizing. If False, the image is resized to the target size without cropping.
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0,0,0), swapRB=True, crop=False)

    # pass blob to the network: This means that the network will perform the inference on the blob data.
    net.setInput(blob)

    # perform predictions:
    objects = net.forward()
    return objects


# display text:
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# x and y are the coordinates of the bottom-left corner of the text string in the image.
def display_text(im, text, x, y):
    # get text size:
    # The cv2.getTextSize() function returns a tuple of two values: the first value is another tuple of two values that represent the width 
    # and height of the text string, and the second value is the baseline of the text string. The baseline is the distance 
    # from the bottom-most point of the text to the bottom of the text box.
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]

    # use text size to create a black rectangle:
    # im: the image on which the rectangle is drawn.
    # (x, y-dim[1] - baseline): the top-left corner of the rectangle. It is calculated by subtracting the height and the baseline of the text string from the y-coordinate of the bottom-left corner of the text string.
    # (x + dim[0], y + baseline): the bottom-right corner of the rectangle. It is calculated by adding the width and the baseline of the text string to the x and y coordinates of the bottom-left corner of the text string.
    # (0,0,0): the color of the rectangle in BGR format.
    # cv2.FILLED: the thickness of the rectangle border in pixels
    cv2.rectangle(im, (x, y-dim[1] - baseline), (x + dim[0], y + baseline), (0,0,0), cv2.FILLED)

    # display text inside the rectangle:
    # im: the image on which the text is put.
    # text: the text string to be put.
    # (x, y-5): the bottom-left corner of the text string in the image. It is slightly above the bottom-left corner of the rectangle to create some margin.
    # cv2.LINE_AA: line will look smoother and less jagged.
    cv2.putText(im, text, (x, y-5), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)




# display objects:
# threshold represents the minimum confidence score required to display an object.
def display_objects(im, objects, threshold=0.25):
    # value represents the height of the image in pixels.
    rows = im.shape[0]
    # value represents the width of the image in pixels.
    cols = im.shape[1]
    # for every detected object:
    # a loop that iterates over the third dimension of the objects array. This dimension represents the number of detected objects in the image.
    for i in range(objects.shape[2]):
        # find the class and confidence
        # assigns the second element, represents the class ID of the detected object, which corresponds to a label in label list.
        classId = int(objects[0, 0, i, 1])
        # assigns the third element, represents the confidence score of the detected object, which ranges from 0 to 1.
        score = float(objects[0, 0, i, 2])

        # recover original coordinates from normalised coordinates
        # assigns the fourth element, represents the x-coordinate of the top-left corner of the bounding box of the detected object in the image.
        x = int(objects[0, 0, i, 3] * cols)
        # assigns the fifth element, represents the y-coordinate of the top-left corner of the bounding box of the detected object in the image.
        y = int(objects[0, 0, i, 4] * rows)
        # assigns the sixth element, represents the width of the bounding box of the detected object in the image.
        w = int(objects[0, 0, i, 5] * cols - x)
        # assigns the seventh element, represents the height of the bounding box of the detected object in the image.
        h = int(objects[0, 0, i, 6] * rows - y)

        # check if the detection is of good quality
        if score > threshold:
            display_text(im, "{}".format(labels[classId]), x, y)
            # (x + w, y+ h): the bottom-right corner of the rectangle. It is calculated by adding the width and the height of the bounding box of the object to the x and y coordinates of the top-left corner of the bounding box.
            # 2: the thickness of the rectangle border in pixels.
            cv2.rectangle(im, (x, y), (x + w, y+ h), (255, 255, 255), 2)

    # convert image to RGB since we're using matplotlib for displaying image.
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(30,10))
    plt.imshow(mp_img)
    plt.show()




# RESULTS:
    
"""im = cv2.imread('images/street.jpg')
objects = detect_objects(net, im)
display_objects(im, objects)
"""


"""im = cv2.imread('images/000000001000.jpg')
objects = detect_objects(net, im)
display_objects(im, objects)"""



"""im = cv2.imread('images/baseball.jpg')
objects = detect_objects(net, im)
display_objects(im, objects)
"""


"""im = cv2.imread('images/giraffe-zebra.jpg')
objects = detect_objects(net, im)
display_objects(im, objects)"""


"""
im = cv2.imread('images/soccer.jpg')
objects = detect_objects(net, im)
display_objects(im, objects)"""




im = cv2.imread('images/teddy.jpg')
objects = detect_objects(net, im)
display_objects(im, objects)



        

