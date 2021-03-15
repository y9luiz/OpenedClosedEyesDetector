pip3 install gdown
sudo apt-get install cmake
sudo apt-get install unrar
if [ ! -d thirdparty ]
then
    mkdir thirdparty
fi
if [ ! -d dataset ]
then
    mkdir dataset
fi
if [ ! -d dataset/eyes ]
then
    mkdir dataset/eyes
    mkdir dataset/eyes/ClosedFace
    mkdir dataset/eyes/OpenFace

fi
if [ ! -d thirdparty/faceDetector ]
then
    git clone https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB thirdparty/faceDetector
fi
if [ ! -d models ]
then
    mkdir models
fi
if [ ! -f models/shape_predictor_68_face_landmarks.dat ]
then
    cd models
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 
    cd ..
fi
if [ ! -f models/eyes_closed_open_model_64_64.h5 ]
then
    cd models
    gdown https://drive.google.com/uc?id=10yUoGbkzXpoIKlhOgtGUmlLqurPDEqgF
    cd ..
fi
if [ ! -d  dataset/dataset_B_FacialImages ]
then
    cd dataset
    gdown https://drive.google.com/uc?id=1niyedvpnATsWMnhcy_DfNNhPGc2J_G8V
    echo "extraindo dataset"
    unrar x dataset_B_Facial_Images.rar -idq
    cd ..
fi

#http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

export PYTHONPATH="./thirdparty/"
