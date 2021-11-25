import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping


def get_data():
    base_path = '/var/lib/vz/data/d12019953/datasets/'
    train = pd.read_csv(base_path + 'digit-recognizer/train.csv')
    test = pd.read_csv(base_path + 'digit-recognizer/test.csv')

    # on sépare les images et leurs labels pour le train
    y_train_local = train["label"]
    x_train_local = train.drop(labels=["label"], axis=1)

    # normalisation des images (val d'un pixel entre 0 et 255)
    x_train_local = x_train_local / 255.0
    x_test_local = test / 255.0

    # mettre les labels sous forme discrète
    y_train_local = to_categorical(y_train_local, num_classes=10)

    # redimensionnement des images pour les injecter dans le réseau
    x_train_local = x_train_local.values.reshape(-1, 28, 28, 1)
    x_test_local = x_test_local.values.reshape(-1, 28, 28, 1)

    print(x_train_local.shape)
    print(y_train_local.shape)
    print(x_test_local.shape)

    return x_train_local, y_train_local, x_test_local


def create_dnn(input_shape=(28, 28, 1), num_classes=10):
    model_local = Sequential()
    model_local.add(Flatten(input_shape=input_shape))
    model_local.add(Dense(512, activation='relu'))
    model_local.add(Dropout(0.2))
    model_local.add(Dense(1024, activation='relu'))
    model_local.add(Dropout(0.2))
    model_local.add(Dense(num_classes, activation='softmax'))

    # on compile avec un optimiseur adam et une loss multi-catégories
    model_local.compile(optimizer="adam",
                        loss="categorical_crossentropy",
                        metrics=["accuracy"])
    return model_local


def create_cnn_model():
    model_local = Sequential()
    #block
    model_local.add(Conv2D(32, kernel_size=3,
                           activation='relu',
                           input_shape=(28, 28, 1)))
    model_local.add(BatchNormalization())
    model_local.add(Conv2D(32, kernel_size=3,
                           activation='relu'))
    model_local.add(BatchNormalization())
    model_local.add(Conv2D(32, kernel_size=5,
                           strides=2,
                           padding='same',
                           activation='relu'))
    model_local.add(BatchNormalization())
    model_local.add(Dropout(0.4))

    # block
    model_local.add(Conv2D(64, kernel_size=3,
                           activation='relu'))
    model_local.add(BatchNormalization())
    model_local.add(Conv2D(64, kernel_size=3,
                           activation='relu'))
    model_local.add(BatchNormalization())
    model_local.add(Conv2D(64, kernel_size=5,
                           strides=2,
                           padding='same',
                           activation='relu'))
    model_local.add(BatchNormalization())
    model_local.add(Dropout(0.4))

    # block
    model_local.add(Conv2D(128, kernel_size=4,
                           activation='relu'))
    model_local.add(BatchNormalization())
    model_local.add(Flatten())
    model_local.add(Dropout(0.4))
    model_local.add(Dense(10, activation='softmax'))

    # on compile avec un optimiseur adam et une loss multi-catégories
    model_local.compile(optimizer="adam",
                        loss="categorical_crossentropy",
                        metrics=["accuracy"])
    return model_local


def create_cnn_model_2():
    pass


if __name__ == '__main__':
    x_train, y_train, x_test = get_data()

    # on définit un générateur d'augmentation de données avec
    # rotation, zoom, décalage en largeur et hauteur
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1)

    # on sépare le train en train/val avec 90/10 pour estimer l'erreur
    # en généralisation
    x_train_, x_val_, y_train_, y_val_ = train_test_split(x_train, y_train,
                                                          test_size=0.1)

    # on réduit le pas de descente du gradient à chaque époque
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

    # on stoppe l'apprentissage si on stagne après un certain nombre
    # d'itérations sans amélioration de notre fonction objectif
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0.00001,
                       patience=10,
                       verbose=1,
                       mode='auto')

    # paramètres conseillés par keras
    nb_epochs = 50
    batch_size = 64
    steps_per_epoch = x_train_.shape[0] // batch_size

    # préparation du flow
    flow = datagen.flow(x_train_, y_train_, batch_size=batch_size)

    # apprentissage !
    # model = create_dnn_model()
    model = create_cnn_model()
    history = model.fit_generator(flow,
                                  epochs=nb_epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=(x_val_, y_val_),
                                  callbacks=[annealer, es],
                                  verbose=1)

    # todo plot les courbes d'apprentissage & les erreurs
    # à distance sur Pycharm et ssh + X (display)

    # prédire pour le test et générer le fichier csv pour la soumission Kaggle
    results = np.zeros((x_test.shape[0], 10))
    results = results + model.predict(x_test)
    results = np.argmax(results, axis=1)
    results = pd.Series(results, name="Label")
    submission = pd.concat(
        [pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
    submission.to_csv("mnist_submission_2.csv", index=False)
