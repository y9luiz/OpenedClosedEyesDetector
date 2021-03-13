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
    # Carregue todas as imagens e as labels
    #       OpenedFace e ClosedFace
    data,labels = loadDataset(DATASET_PATH)
    # Transforme as labels em numeros
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    # "split" o dataset em 2 conjuntos, um de teste
    # e outro de treino, onde 20% das imagens totais são
    # imagens de teste e o restante de treino
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)
    # Geraremos dados de treino e de teste aplicando
    # transformações nas imagens de treino e teste
    # para deixar nossa rede mais precisa
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    # utilizando o modelo pretreinado mobilinetV2
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(64, 64, 3)))
    # Adicionamos mais algumas camadas adicionais
    headModel = baseModel.output
    # reduzimos a saída do nosso modelo
    headModel = MaxPool2D(pool_size=(2, 2))(headModel)
    # Transformamos uma matriz em array
    headModel = Flatten(name="flatten")(headModel)
    # adicionamos dense layers para e prever a imagem
    headModel = Dense(128, activation="relu")(headModel)
    # Adicionamos também uma camada de dropout na
    # tentativa de evitar overfiting
    headModel = Dropout(0.5)(headModel)
    # ultima camada que irá definir se temos uma
    # OpenedFace ou uma Closed Face
    headModel = Dense(2, activation="softmax")(headModel)
    # Juntamos tudo em um unico model
    model = Model(inputs=baseModel.input, outputs=headModel)
    # mostramos suas camadas
    print(model.summary())
    # marcamos para treino as 8 ultimas camadas do modelo 
    # da Mobilinetv2, ou seja, um fining tuning
    for layer in baseModel.layers[8:]:
    	layer.trainable = True
    # Aplicamos o adam optimizer
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # Como temos somente 2 labels a binary_crossentropy
    # é uma boa escolha
    model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])
    print("Train started")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS, 
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS, 
        epochs=EPOCHS)
    # Para cada imagem teste precisamos encontrar a label que melhor
    # casa com ela
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)
    # printamos os resultados baseado nas predições feitas no
    # dataset de teste
    print(classification_report(testY.argmax(axis=1), predIdxs,
        target_names=lb.classes_))
    print("salvando o modelo")
    model.save("models/eyes_detector_model.h5", save_format="h5")
    # plot dos dados
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