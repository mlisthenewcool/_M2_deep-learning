from IPython.display import display
import numpy as np
import json
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

def show_images(images, cols=1):
    n_images = len(images)
    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        plt.axis('off')
        if image.ndim == 2:
            plt.gray()
        plt.savefig(image)


'''
    # DÃ©finition d'un modÃ¨le via la "graph" API de Keras
    model_tmp = Model(model_cnn_1.layers[0].input, model_cnn_1.layers[0].output)

    # Cartes rÃ©ponses de x_test[10] (un '0')
    feature_maps = model_tmp.predict(np.expand_dims(x_test[10], 0))

    # normalisation des valeurs entre 0 et 1
    minimum, maximum = np.min(feature_maps), np.max(feature_maps)
    feature_maps = (feature_maps - minimum) / (maximum - minimum)

    print(minimum, maximum, feature_maps.shape)

    images = []
    for i in range(64):
        images.append(np.array(255 * feature_maps[:,:,:,i])
                      .reshape(28, 28).astype('uint8'))

    show_images(images, 8)

    """Les images obtenues semblent montrer que les filtres appris dans la 
    premiÃ¨re couche du rÃ©seau permettent la dÃ©tection d'orientations ou 
    des contours.

    <h4>Visualisation des filtres</h4>

    La visualisation des filtres de la couche d'entrÃ©e est relativement 
    immÃ©diate puisque que les filtres ont le mÃªme nombre de channels que les
     donnÃ©es d'entrÃ©e. Autrement dit, les filtres de la premiÃ¨re couche sont 
     dÃ©finis sur le mÃªme domaine que les donnÃ©es.
    """

    # RÃ©cupÃ¨re tous les paramÃ¨tres appris par le rÃ©seau
    weights = model_cnn_1.get_weights()
    [print(w.shape) for w in weights]

    # normalize values between 0 and 1
    minimum, maximum = np.min(weights[0]), np.max(weights[0])
    weights0 = (weights[0] - minimum) / (maximum - minimum)

    print(minimum, maximum)
    # entre 0 et 255 pour l'affichage
    weights0 *= 255.

    images = []
    for i in range(64):
        images.append(np.array(255*weights0[:,:,:,i])
                      .reshape(5,5).astype('uint8'))

    show_images(images, 8)

    """<h3> CNN pour des images naturelles </h3>

    Les filtres de convolutions appris sur les donnÃ©es MNIST ne sont pas trÃ¨s
     parlantes, intÃ©ressons nous plutÃ´t Ã  un rÃ©seau (beaucoup plus profond) 
     adaptÃ© Ã  la classification d'images naturelles. Nous allons considÃ©rer 
     pour cela le réseau VGG16, pré-entrainé sur le dataset ImageNet (cf cours).
    """

    # !wget https://pageperso.lis-lab.fr/stephane.ayache/cat.jpg
    # !wget https://pageperso.lis-lab.fr/stephane.ayache/imagenet_class_index.json
    '''

# RÃ©cupÃ©ration du modÃ¨le complet VGG16
model3 = VGG16(include_top=True, weights='imagenet')
# print(model3.summary())

# chargement d'un dictionnaire qui met en correspondant l'index d'une classe ImageNet et son nom
with open('imagenet_class_index.json') as f:
    CLASS_INDEX = json.load(f)

# chargement et preprocess de l'image (dont normalisation et redimensionnement)
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# prÃ©diction et affichage de la classe de probabilitÃ© maximale
softmax_output = model3.predict(x)
best_class = np.argmax(softmax_output)
im_class = CLASS_INDEX[str(best_class)][1]
print("prediction: ", im_class)

"""Notre chaton a Ã©tÃ© reconnu comme un chat ? 
Alors tout va bien, et regardons les filtres de la premiÃ¨re couche ..."""

weights = model3.get_weights()
# for w in weights: print(w.shape)

# normalize values between 0 and 1
minimum, maximum = np.min(weights[0]), np.max(weights[0])
weights0 = (weights[0] - minimum) / (maximum - minimum)

print(minimum, maximum)
# entre 0 et 255 pour l'affichage
weights0 *= 255.

images = []
for i in range(64):
    images.append(
        np.array(255 * weights0[:, :, :, i]).reshape(3, 3, 3).astype('uint8'))

show_images(images, 8)

"""Notez qu'on affiche ici les fltres en RGB alors qu'on pourrait les 
afficher en sÃ©parant les channels. Dans tous les cas, cette visualisation
reste peu informative, pour mieux comprendre le rÃ´le d'une couche de
convolution, d'autres mÃ©thodes sont plus efficaces (cf cours prÃ©cÃ©dent).
Ci dessous, nous allons gÃ©nÃ©rer une image qui maximise la valeur d'une
carte d'activation (rÃ©ponse Ã  un filtre).

<h3> Visualisation par maximisation d'activations</h3>

Les fonctions Tensorflow du backend Keras nous permettent de dÃ©finir une
fonction d'une image vers les activations d'une carte d'une couche de
convolution. Maximiser cette fonction en suivant le gradient de la sortie
par rapport aux entrÃ©es, revient Ã  dÃ©terminer une image (artificielle)
dont VGG16 obtient une forte rÃ©ponse Ã  la convolution du filtre ciblÃ©.
L'image produite va Ãªtre composÃ©e du "pattern" reconnu par le filtre...

D'abord, quelques fonctions utiles :
"""


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    return x * 255.


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


print(model3.summary())

"""DÃ©finition d'une fonction en Keras ; obtention du gradient ; 
gradient ascent pour maximiser ;"""

from keras import backend as K

num_filter = 208

# entrÃ©e de la fonction Ã  maximiser = entrÃ©e du rÃ©seau VGG = une image
input_img = model3.layers[0].input

# sortie = Somme (moyenne) des activations d'une carte de convolution
# associÃ©e au filtre num_filter
output = K.mean(model3.layers[17].output[:, :, :, num_filter])

# returns the gradients of output w.r.t. input
grads = K.gradients(output, input_img)[0]
grads = normalize(grads)

# fonction : input -> output, gradients
func = K.function([input_img], [output, grads])

## gradient ascent ##

# point de dÃ©part alÃ©atoire
x = np.random.random((1, 224, 224, 3)) * 255.
x = preprocess_input(x)

for i in range(40):  # 40 iterations
    loss_value, grads_value = func([x])
    x += grads_value * 10

    print('Current loss value:', loss_value)
    if loss_value <= 0.:
        # some filters get stuck to 0
        break

# point d'arrivÃ©e : conversion en RGB puis affichage
xx = deprocess_image(x)

# from PIL import Image
print(xx.shape)
# display(Image.fromarray(xx[0].astype('uint8')))

"""<b> A faire : </b> variez les couches et les filtres, puis commentez vos 
rÃ©sultats obtenus. Comment obtenir des motifs encore plus complexes ?"""

"""   
<h2> Entrainement d'un rÃ©seau trÃ¨s profond </h2>

Un rÃ©seau trÃ¨s profond tel que VGG contient Ã©normÃ©ment de paramÃ¨tres,
son entrainement peu s'avÃ©rer compliquer. Tous les paramÃ¨tres doivent 
entrer en mÃ©moire (RAM ou GPU), ainsi que toutes les donnÃ©es d'un 
minibatch. Dans le cas d'images relativement volumineuses 
(ie: Imagenet =255x255x3), la taille du minibatch est souvent ainsi trÃ¨s 
couteux en mÃ©moire et se voit ainsi rÃ©duit Ã  quelques images, 
rendant l'entrainement encore plus long... 

Souvent, de grandes bases de donnÃ©es ne sont pas disponibles. Dans ce cas,
 pour parvenir Ã  entrainer un tel rÃ©seau avec peu d'images, 
 plusieurs options sont possibles :
- partir d'un rÃ©seau dÃ©jÃ  (prÃ©-)entrainÃ©, et/ou utiliser ses poids 
pour initialiser un autre rÃ©seau qui sera appris sur une base (rÃ©duite) 
d'images (= finetuning, trÃ¨s efficace)
- rÃ©gulariser les poids du rÃ©seau (peut aider mais ne suffit pas)
- augmenter artificiellement le nombre de donnÃ©es (amÃ©liore toujours 
les performances d'un rÃ©seau)

Dans la suite, nous illustrons ce dernier point sur les donnÃ©es MNIST.
"""

"""La performance obtenue peut Ãªtre amÃ©liorÃ©e en augmentant les donnÃ©es.
 La cellule suivante gÃ©nÃ¨re 20000 donnÃ©es Ã  partir des 5000 utilisÃ©es 
 jusque lÃ ."""
