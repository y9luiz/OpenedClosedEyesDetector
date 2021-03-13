import sys
from imutils import paths,face_utils
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from src.face_landmark_predictor import FaceLandmarkPredictor

def getDatasetImages(dataset_path):
    imagePaths = list(paths.list_images(dataset_path))
    data = []
    labels = []
    filenames = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-2]
        filename = imagePath.split(os.path.sep)[-1]
        data.append(image)
        labels.append(label)
        filenames.append(filename)
    return data,labels,filenames

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
    if len(sys.argv)!=2:
        raise IOError("usage: python3 preprocess_dataset.py dataset_path")
    img_list,labels,files = getDatasetImages(sys.argv[1])
    face_landmark_predictor = FaceLandmarkPredictor()
    for img,label,file in zip(img_list,labels,files):
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        landmark_list = face_landmark_predictor.predictLandmarks(img_gray,[0,0,100,100])        
        isInRange = lambda idx, idx_range :  True if idx>=idx_range[0] and idx<=idx_range[1] else False
        range_idxs = [] 
        range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"])
        range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"])
        range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["right_eyebrow"])
        range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["left_eyebrow"])
        good_landmarks_list = []
        for i in range(48):
            for range_idx in range_idxs:
                if isInRange(i, range_idx) or i == 30:
                    good_landmarks_list.append(list(landmark_list[i]))
        bb = getBoudingboxFromLandmarksList(good_landmarks_list)
        pt1 = (bb[0][0],bb[0][1])
        pt2 = (bb[1][0],bb[1][1])
        try:
            eye_img = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
            eye_img = cv2.resize(eye_img,64,64),interpolation=cv2.INTER_AREA)
        except:
            continue
        new_file_path =f'./dataset/eyes/{label}/{file}' 
        cv2.imwrite(new_file_path,eye_img)

