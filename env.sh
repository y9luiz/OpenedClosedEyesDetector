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
#http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

export PYTHONPATH="./thirdparty/"
