from src.face_detector import FaceDetector
from src.face_landmark_predictor import FaceLandmarkPredictor

import sys
import cv2
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    face_landmark_predictor = FaceLandmarkPredictor()
    while cap.isOpened:
        ret, img = cap.read()
        if not ret:
            IOError("Camera device error")
            break
        for bbox in face_detector.detectFaces(img):
            pt1 = (bbox[0],bbox[1])
            pt2 = (bbox[2],bbox[3])
            cv2.rectangle(img,pt1,pt2,(255,0,0))
            face_img = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
            landmark_list = face_landmark_predictor.predictLandmarks(face_img,bbox)
            for landmark in landmark_list:
                cv2.circle(img,landmark,1,(0,255,0))
        cv2.imshow("frame", img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()