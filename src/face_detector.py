import cv2
from cv2 import dnn
from faceDetector.caffe import ultra_face_opencvdnn_inference as ultra_face
import numpy as np
import time
import dlib

MODEL_PATH = "./thirdparty/faceDetector/caffe/model/Slim-320/slim-320.caffemodel"
PROTOFILE_PATH = "./thirdparty/faceDetector/caffe/model/Slim-320/slim-320.prototxt"
LANDMARK_PREDICTOR_PATH="./models/shape_predictor_68_face_landmarks.dat"
INPUT_HEIGHT = 240
INPUT_WIDTH = 320

class FaceDetector:
    def __init__(self,net=None, threshold = 0.8):
      if type(net) != None:
        self.__net__ = net
      self.__net__ = dnn.readNetFromCaffe(PROTOFILE_PATH, MODEL_PATH)
      self.__priors__ = ultra_face.define_img_size([INPUT_WIDTH,INPUT_HEIGHT])
      self.__threshold__ = threshold
    def detectFaces(self,image):
        self.__net__.setInput(dnn.blobFromImage(image, 1 / 128.0, (INPUT_WIDTH, INPUT_HEIGHT), 127))
        boxes, scores = self.__net__.forward(["boxes", "scores"])
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        boxes = ultra_face.convert_locations_to_boxes(boxes, self.__priors__, ultra_face.center_variance, ultra_face.size_variance)
        boxes = ultra_face.center_form_to_corner_form(boxes)
        boxes, labels, probs = ultra_face.predict(image.shape[1], image.shape[0], scores, boxes, ultra_face.args.threshold)     
        good_boxes = []
        for i in range(boxes.shape[0]):
            if probs[i]>self.__threshold__:
                box = boxes[i,:]
                good_boxes.append(box)  
        return good_boxes  
def convert2InputImage(image):
    """
        get an image and convert it to the neural network input
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    return image