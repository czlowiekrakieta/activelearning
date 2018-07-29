from glob import glob
import os, pickle, numpy as np
import matplotlib.pyplot as plt

DATASETS = '/home/luke/DATASETS/'


def normalize(X):
    return X.astype(float)/255.


def read_batch_small(nr=1):
    bpath = '/home/luke/DATASETS/cifar-10-batches-py/data_batch_{}'.format(nr)
    with open(bpath, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def process_cifar10_dict(data_dict):
    """
    return X, y, X beings images in format (-1, 32, 32, 3), while y are labels in form of digits

    :param data_dict:
    :return:
    """
    X = data_dict[b'data']
    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = data_dict[b'labels']
    return normalize(X), y


def read_cifar10_data(base=DATASETS):
    x_all = []
    y_all = []
    for i in range(5):
        dat = read_batch_small(i+1)
        x, y = process_cifar10_dict(dat)
        x_all.append(x)
        y_all.extend(y)

    return np.concatenate(x_all, axis=0), np.array(y_all)


def read_cifar100_data(base=DATASETS):
    """
    returns X, y_fine, y_coarse
    X are images in format (-1, 32, 32, 3)
    y_fine are fine labels (detailed)
    y_coars are coarse labels (general category)

    :param base:
    :return:
    """
    dat = glob(os.path.join(base, 'cifar-100-python', '*'))
    dat = pickle.load(open(dat[-1], 'rb'), encoding='bytes')
    X = dat[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_coarse = dat[b'coarse_labels']
    y_fine = dat[b'fine_labels']
    return normalize(X), y_fine, y_coarse


def read_caltech_101(track=False):
    """
    returns X, y, names
    X is list of images (NOTE: not numpy ndarray)
    y is list of labels in form od digits
    names is list of names assigned to labels, in the same order as digits in y variable

    :param track:
    :return:
    """

    cats_folders = glob(os.path.join(DATASETS, '101_ObjectCategories', '*'))
    cats_names = [x.split('/')[-1] for x in cats_folders]
    X = []
    y = []
    cnt = 0
    names = list()
    for name, folder in zip(cats_names, cats_folders):
        if track:
            print(cnt, name)

        if name not in names:
            names.append(name)
        imgs = glob(os.path.join(folder, '*.jpg'))
        L = []
        for i in imgs:
            z = plt.imread(i)
            X.append(z)
            y.append(cnt)
        cnt += 1

    return X, y, names
