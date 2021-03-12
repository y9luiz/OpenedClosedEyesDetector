from tensorflow.keras.preprocessing.image import ImageDataGenerator                                                                                                                             
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D,MaxPool2D, Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import os
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 

def loadDataset(input_path):
    imagePaths = list(paths.list_images(input_path))
    data = []
    labels = []
    preprocess = lambda img: img_to_array(img)/255.0
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = load_img(imagePath, target_size=(64, 64))
        image = preprocess(image)
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    return (data,labels)

INIT_LR = 1e-4
EPOCHS = 40
BS = 32
DATASET_PATH = "./dataset/eyes/"
if __name__=="__main__":
    data,labels = loadDataset(DATASET_PATH)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(64, 64, 3)))
    #print(dir(baseModel.summary))
    #exit(0)
    headModel = baseModel.output
    headModel = MaxPool2D(pool_size=(2, 2))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    #headModel = Dense(512, activation="relu")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    #headModel = Dense(32, activation="relu")(headModel)

    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    print(model.summary())
    for layer in baseModel.layers[8:]:
    	layer.trainable = True
    #for layer in baseModel.layers:
    #    layer.trainable = False
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])
    # train the head of the network
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS, 
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS, 
        epochs=EPOCHS)

    predIdxs = model.predict(testX, batch_size=BS)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,
        target_names=lb.classes_))
    # serialize the model to disk
    print("[INFO] saving mask detector model...")
    model.save("models/eyes_detector_model.h5", save_format="h5")
    # plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot( H.history["loss"], label="train_loss")
    print(H.history.keys())
    for key in H.history.keys():
        print(f'key:{key} {H.history[key]}')    
    plt.plot( H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot_novo")     