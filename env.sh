if [ ! -d thirdparty ]
then
    mkdir thirdparty
fi
if [ ! -d thirdparty/faceDetector ]
then
    git clone https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB thirdparty/faceDetector
fi

export PYTHONPATH="./thirdparty/"
