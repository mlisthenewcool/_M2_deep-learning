import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import (
    Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization,
    SeparableConv2D)
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.applications.vgg19 import VGG19
from sklearn.model_selection import train_test_split


def get_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # on convertit les labels en des matrices binaires
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    return x_train, y_train, x_test, y_test


def create_cnn(input_shape=(32, 32, 3), num_classes=10):
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


def create_cnn_custom(input_shape=(32, 32, 3),
                      num_classes=10,
                      do_transfer_learning=False):
    # build vgg 19 model
    num_layers_from_vgg = 11

    vgg = VGG19(input_shape=input_shape,
                weights='imagenet',
                include_top=False)
    print('Load VGG19 as base model')

    # block 4
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_sepconv1')(
        vgg.layers[num_layers_from_vgg].output)
    x = BatchNormalization(name='block4_conv1_bn')(x)
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_conv2_bn')(x)
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block4_sepconv3')(x)
    x = MaxPooling2D((2, 2), name='block4_pool')(x)

    # block 5
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_sepconv1')(x)
    x = BatchNormalization(name='block5_conv1_bn')(x)
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_sepconv2')(x)
    x = BatchNormalization(name='block5_conv2_bn')(x)
    x = SeparableConv2D(512, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block5_sepconv3')(x)
    x = MaxPooling2D((2, 2), name='block5_pool')(x)

    # adding classification block on top
    x = Flatten(input_shape=input_shape, name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # combine both
    model = Model(inputs=vgg.input, outputs=x, name='Custom CNN')

    # load top model weights
    # model.load_weights(top_model_weights_path)

    # freeze layers
    if do_transfer_learning:
        for i, layer in enumerate(model.layers):
            if i <= num_layers_from_vgg:
                layer.trainable = False
            else:
                layer.trainable=True

    # on compile avec un optimiseur adam et une loss multi-catégories
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_data()

    # on sépare le train en train/val pour estimer l'erreur en généralisation
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

    # paramètres d'apprentissage pour toutes les expériences
    nb_epochs = 200
    batch_size = 64

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
    model = create_cnn()

    # entrainement avec le CNN établi lors du TER
    # model = create_cnn_custom()

    print("[CNN WITH DATA AUGMENTATION] Architecture")
    print(model.summary())

    history = model.fit_generator(flow,
                                  epochs=nb_epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=(x_val_, y_val_),
                                  callbacks=[annealer, es],
                                  verbose=1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("[CNN WITH DATA AUGMENTATION] Score : {}".format(score))

    model.save('model_cnn_cifar.h5')
