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


# dnn module allows us to run deep learning inference on images and videos using diff models and frameworks.
# creates a network object by reading the model files from the disk using cv2.dnn.readNetFromCaffe() function. Takes two arguments:
# 1. 'deploy.prototxt' file which contains the network architecture and the layer parameters of the model.
# 2. "res10_300x300_ssd_iter_140000_fp16.caffemodel" contains the weights and biases of the model.
net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                               "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model Parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7


# ASCII code for esc key is 27. 
while cv2.waitKey(1) != 27:
    #  The has_frame variable stores the boolean value and the frame variable stores the array.
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # create a 4D blob from a frame:
    # A blob is a 4-dimensional array that contains the image data after some preprocessing steps.
    # frame is the input image that we want to convert into a blob.
    # 1.0 is the scale factor that multiplies the image pixel values. It is used to normalize the pixel values to a certain range, such as 0-1 or 0-255.
    # (in_width, in_height) is the target size that we want to resize the image to. It is usually the same as the input size of the neural network that we want to use for inference.
    # mean is a list of three values that are subtracted from each channel of the image. It is used to center the pixel values around zero and reduce the effect of illumination changes.
    # swapRB = a boolean flag that indicates whether to swap the red and blue channels of the image.
    # crop = a boolean flag that indicates whether to crop the image from the center. If True, the image is cropped to the target size without resizing. If False, the image is resized to the target size without cropping.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=True)

    # run a model:
    net.setInput(blob)
    # runs a forward pass to compute the output of the network. The output is a numpy array that contains the predictions of the
    # network, such as confidence scores and the bounding boxes for the detected object.
    detections = net.forward()
    
    # loops over the number of detected faces in the output array of the network. The output array has the shape  of (1, 1, N, 7)
    # where N is the number of faces and 7 is the number of values for each face.
    for i in range(detections.shape[2]):
        # gets the confidence score of the i-th face from the output array.
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)
            
            # bounding box corners, color and thickness.
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (255, 0, 0), thickness = 1)
            label = "Confidence: %.4f" %confidence
            # text label's font, scale and thickness.
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # rectangle serves as a background for the label.
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                                 (x_left_bottom + label_size[0], y_left_bottom + base_line),
                                 (255, 255, 255), cv2.FILLED)
            # puts the text label on the input image with the specified font, scale, color, and position.
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # gets the total time and the time of each layer of the network inference. 
    t, _ = net.getPerfProfile()
    # creates a string that contains the inference time of the network in milliseconds.
    label = 'Inference Time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    # puts the text label on the input image with the specified font, scale, color, and position. Position is top left corner of the image.
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)


source.release()
#The cv2.destroyWindow function destroys the specified window and frees the memory associated with it.
cv2.destroyWindow(win_name)





