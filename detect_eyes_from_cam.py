import cv2                                                                                                                                                                                      
from cv2 import dnn
from faceDetector.caffe import ultra_face_opencvdnn_inference as ultra_face
import numpy as np
import time
import dlib
from imutils import face_utils
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from src.face_detector import FaceDetector
from src.face_landmark_predictor import FaceLandmarkPredictor
import time
def getBoudingboxFromLandmarksList(landmaks_list):
    bouding_box = [landmaks_list[0],landmaks_list[-1]]
    for landmark in landmaks_list:
        if bouding_box[0][0] < bouding_box[0][0]:
            bouding_box[0][0] = landmark[0]
        if landmark[1] < bouding_box[0][1]:
            bouding_box[0][1] = landmark[1]
        if landmark[0] > bouding_box[1][0]:
            bouding_box[1][0] = landmark[0]
        if landmark[1] > bouding_box[1][1]:
            bouding_box[1][1] = landmark[1] 
    return bouding_box
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    face_landmark_predictor = FaceLandmarkPredictor()
    range_idxs = [] 
    range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"])
    range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"])
    range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["right_eyebrow"])
    range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["left_eyebrow"])
    model = load_model("./test_eyes2_novo.h5")
    while cap.isOpened:
        ret, img = cap.read()
        t_inicial = time.time()
        img = cv2.resize(img,(240,320))
        show_img = img.copy()
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        for box in face_detector.detectFaces(img):
            landmark_list = face_landmark_predictor.predictLandmarks(img_gray,box) 
       
            isInRange = lambda idx, idx_range :  True if idx>=idx_range[0] and idx<=idx_range[1] else False
            good_landmarks_list = []
            for i in range(48):
                for range_idx in range_idxs:
                    if isInRange(i, range_idx) or i == 30:
                        good_landmarks_list.append(list(landmark_list[i]))
                        #cv2.circle(show_img,landmark_list[i],1,(255,0,0))
            bb = getBoudingboxFromLandmarksList(good_landmarks_list)

            pt1 = (bb[0][0],bb[0][1])
            pt2 = (bb[1][0],bb[1][1])
            eye_img = None
            try:
                eye_img = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
                eye_img = cv2.resize(eye_img,(64,64),interpolation=cv2.INTER_AREA)
            except:
                continue

            eye_img = eye_img/255
            eye_img = img_to_array(eye_img)
            eye_img = np.expand_dims(eye_img, axis=0)

            (closed, opened) = model.predict(eye_img)[0]
            time_elapsed = (time.time()-t_inicial)*1000

            print(f'tempo gasto: {time_elapsed}')
            #print(f'closed score: {closed} , opened score: {opened}')
            label = f'Closed: {closed*100}'
            score = closed
            color = (0,0,255)
            if opened > closed:
                label = f'Opened: {opened*100}'
                score = opened
                color = (0,255,0)
            if score>0.9:
                cv2.putText(show_img,label,(int(box[2]/2),box[3]),1,1,color,thickness=1)     
                print(label) 
        cv2.imshow("img",show_img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()
