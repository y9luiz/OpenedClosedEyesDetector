import dlib
LANDMARK_MODEL_PREDICTOR_PATH = "./models/shape_predictor_68_face_landmarks.dat"
class FaceLandmarkPredictor:
    def __init__(self):
        self._predictor =  dlib.shape_predictor(LANDMARK_MODEL_PREDICTOR_PATH)
        self._n_landmarks = 68
    def predictLandmarks(self,face_img, bounding_box):
        facial_landmarks_list = []
        d_rect = dlib.rectangle(bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3])
        shape = self._predictor(face_img,d_rect)
        for i in range(self._n_landmarks):
            facial_landmarks_list.append((shape.part(i).x,shape.part(i).y))
        return facial_landmarks_list
