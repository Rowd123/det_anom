import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torchgen.api.cpp import return_type


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):

    '''
    Cette fonction a pour objectif le décalage du centre de la carte de couleurs.
    Utile quand on a des valeurs negatives et d'autres positives et on veut que
    la médiane de la gamme dynamique de couleurs corresponde à la valeur 0

    :param cmap: la 'colormap' de matplotlib utilisée
    :param start: l'offset du plus bas point de colormap,
                Par défaut égale 0.0. Doit être entre 0.0 et midpoint
    :param midpoint: Le nouveau centre de la colormap. Par défaut 0.5
                Doit être enbtre 0.0 et 1.0. En général, il est choisi
                selon la relation 1 - vmax/(vmax+ abs(vmin))
    :param stop: offset du plus haut point de la colormap,
                Par défaut égale à 1.0. Doit être entre 1.0 et midpoint
    :return:
    '''
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    reg_index = np.linspace(start, stop, 257)

    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r ,g ,b ,a = cmap(ri)
        cdict['red'].append([si, r, r])
        cdict['green'].append([si, g, g])
        cdict['blue'].append([si, b, b])
        cdict['alpha'].append([si, a, a])

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

