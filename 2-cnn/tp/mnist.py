import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


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

    # redimensionnement des images pour les injecter dans le réseau
    x_train_local = x_train_local.values.reshape(-1, 28, 28, 1)
    x_test_local = x_test_local.values.reshape(-1, 28, 28, 1)

    # mettre les labels sous forme discrète
    y_train_local = to_categorical(y_train_local, num_classes=10)

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


def create_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    :param input_shape: Taille de l'image en entrée
    :param num_classes: Nombre de classes à prédire
    :return: Un modèle Sequential de Keras
    """
    model_local = Sequential()
    model_local.add(Conv2D(64, kernel_size=(5, 5),
                           strides=(1, 1), padding='same',
                           activation='relu', input_shape=input_shape))
    model_local.add(MaxPooling2D((2, 2)))
    model_local.add(Conv2D(128, kernel_size=(3, 3),
                           strides=(1, 1), padding='same',
                           activation='relu'))
    model_local.add(MaxPooling2D((2, 2)))
    model_local.add(Conv2D(128, kernel_size=(3, 3),
                           strides=(1, 1), padding='same',
                           activation='relu'))
    model_local.add(MaxPooling2D((2, 2)))
    model_local.add(Flatten())
    model_local.add(Dropout(0.2))
    model_local.add(Dense(num_classes, activation='softmax'))

    # on compile avec un optimiseur adam et une loss multi-catégories
    model_local.compile(optimizer="adam",
                        loss="categorical_crossentropy",
                        metrics=["accuracy"])
    return model_local


if __name__ == '__main__':
    x_train, y_train, x_test = get_data()

    # on sépare le train en train/val pour estimer l'erreur en généralisation
    x_train_, x_test_, y_train_, y_test_ = train_test_split(x_train, y_train,
                                                            test_size=0.2)


    # on réduit le pas de descente du gradient à chaque époque
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

    # on stoppe l'apprentissage si on stagne après un certain nombre
    # d'itérations sans amélioration de notre fonction objectif
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0.00001,
                       patience=10,
                       verbose=1,
                       mode='auto')

    # paramètres d'apprentissage pour toutes les expériences
    nb_epochs = 50
    batch_size = 64

    # apprentissage de notre DNN initial !
    model_dnn = create_dnn()
    history = model_dnn.fit(x_train_, y_train_,
                            batch_size=batch_size,
                            epochs=nb_epochs,
                            validation_split=0.1,
                            callbacks=[annealer, es],
                            verbose=0)

    # apprentissage de notre CNN initial !
    model_cnn_1 = create_cnn()
    history = model_cnn_1.fit(x_train_, y_train_,
                              batch_size=batch_size,
                              epochs=nb_epochs,
                              validation_split=0.1,
                              callbacks=[annealer, es],
                              verbose=0)

    # on définit un générateur d'augmentation de données avec :
    # rotation, zoom, décalage en largeur et hauteur
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

    # on instancie notre générateur
    steps_per_epoch = x_train_.shape[0] // batch_size
    datagen.fit(x_train_)
    flow = datagen.flow(x_train_, y_train_, batch_size=batch_size)

    # entrainement du même CNN avec augmentation de données !
    model_cnn_2 = create_cnn()
    history = model_cnn_2.fit_generator(flow,
                                        epochs=nb_epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        callbacks=[annealer, es],
                                        verbose=0)

    # affichage des résultats !
    print("[1 - DNN] Architecture")
    print(model_dnn.summary())
    score_dnn = model_dnn.evaluate(x_test_, y_test_, verbose=0)
    print("[1 - DNN] Score : {}".format(score_dnn))

    print("[2 - CNN BASE] Architecture")
    print(model_cnn_1.summary())
    score_cnn_1 = model_cnn_1.evaluate(x_test_, y_test_, verbose=0)
    print("[2 - CNN BASE] Score : {}".format(score_cnn_1))

    print("[3 - CNN WITH DATA AUGMENTATION] Architecture")
    print(model_cnn_2.summary())
    score_cnn_2 = model_cnn_2.evaluate(x_test_, y_test_, verbose=0)
    print("[3 - CNN WITH DATA AUGMENTATION] Score : {}".format(score_cnn_2))

    # sauvegarde des models
    model_dnn.save('model_dnn.h5')
    model_cnn_1.save('model_cnn_1.h5')
    model_cnn_2.save('model_cnn_2.h5')
