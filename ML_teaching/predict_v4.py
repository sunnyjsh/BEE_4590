# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import sys
from threading import Timer
import tflite_runtime.interpreter as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
import cv2
import helper
from time import time

MODEL_FILENAME = 'model.tflite'
LABELS_FILENAME = 'labels.txt'


class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""
    def __init__(self, model_filename, labels):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tf.Interpreter(model_path=model_filename)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()
        
        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


# Load labels
with open(LABELS_FILENAME, 'r') as f:
    labels = [l.strip() for l in f.readlines()]

od_model = TFLiteObjectDetection(MODEL_FILENAME, labels)

capture=int(1)
# the time image shows on the screen (ms)
t_show = 2000

# ret, frame = video_object.read()
frame = cv2.imread('test.jpeg')
cv2.imshow('frame', frame)

# Update orientation based on EXIF tags, if the file has orientation info.
image = helper.update_orientation(frame)
    
# If the image has either w or h greater than 1600 we resize it down respecting
# aspect ratio such that the largest dimension is 1600
image = helper.resize_down_to_1600_max_dim(image)

# We next get the largest center square
h, w = image.shape[:2]
min_dim = min(w,h)
max_square_image = helper.crop_center(image, min_dim, min_dim)

# Resize that square down to 512x512
augmented_image = helper.resize_to_512_square(max_square_image)

predictions = od_model.predict_image(Image.fromarray(augmented_image))
    
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.imwrite('ML' + str(capture) + '.jpg', augmented_image)
capture=capture+1

# Looping through number of predictions
for pred in predictions:
    if pred['probability'] >= 0.5:
        # Draw rectangle for each bounding box based on left, top pixel + width and height
        topleft = (int(pred['boundingBox']['left'] * augmented_image.shape[0]), int(pred['boundingBox']['top'] * augmented_image.shape[1]))
        bottomright = (int(topleft[0] + pred['boundingBox']['width'] * augmented_image.shape[0]), int(topleft[1] + pred['boundingBox']['height'] * augmented_image.shape[0]))
        print(topleft)
        print(bottomright)
        
        # text to put
        text=f"{pred['tagName']} | {round(pred['probability'] * 100, 2)}%" 
        
        # draw rectangle and text on img
        cv2.rectangle(augmented_image, topleft, bottomright, (255, 0 ,0), 2)
        cv2.putText(augmented_image, text, topleft, font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

cv2.imwrite('ML' + str(capture) + '.jpg', augmented_image)
capture=capture+1    
cv2.imshow('Frames',augmented_image)
cv2.waitKey(t_show)               
