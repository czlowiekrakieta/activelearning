from core import CEAL, UncertaintySampling, Random
from readers import read_caltech_101, read_cifar10_data, read_cifar100_data, read_mnist
from models import smallnet, EarlyStopping

import numpy as np
import argparse

algodict = {
    'CEAL': CEAL,
    'US': UncertaintySampling,
    'random': Random
}

readerdict = {
    'CIFAR10': read_cifar10_data,
    'CIFAR100': read_cifar100_data,
    'CALTECH101': read_caltech_101,
    'MNIST': read_mnist
}

models = {
    'smallnet': smallnet
}


def main():
    intro_parser = argparse.ArgumentParser()
    intro_parser.add_argument('--algorithm', choices=['CEAL', 'random', 'US'], required=True)
    intro_parser.add_argument('--model_name', type=str, required=True)
    intro_parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'CALTECH101', 'MNIST'], required=True)
    intro_parser.add_argument('--labeled_prop', type=float, default=0.1)

    algo_parser = argparse.ArgumentParser()
    algo_parser.add_argument('--method', choices=['en', 'lc', 'ms'], default='en')
    algo_parser.add_argument('--candidate_batch', type=int, default=320)
    algo_parser.add_argument('--task', type=str, choices=['classification', 'regression'], default='classification')
    algo_parser.add_argument('--final_batch', type=int, default=32)
    algo_parser.add_argument('--dump_history_every_k_iterations', type=int, default=30)
    algo_parser.add_argument('--iterations', type=int, default=100)
    algo_parser.add_argument('--mean_to_variance_threshold', default=0.05, type=float)
    algo_parser.add_argument('--entropy_threshold', default=0.05, type=float)
    algo_parser.add_argument('--update_every_k_iterations', default=5, type=int)
    algo_parser.add_argument('--validation_split')

    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--epochs', default=35)
    model_parser.add_argument('--batch_size', default=64)
    model_parser.add_argument('--filters', type=int, nargs='+')
    model_parser.add_argument('--kernels', type=int, nargs='+')
    model_parser.add_argument('--patience', type=int, default=3)
    # model_parser.add_argument('--sampled_epochs', default=2)

    I = intro_parser.parse_known_args()[0]

    X, y = readerdict[I.dataset]()
    sep = int(X.shape[0]*I.labeled_prop)
    labeled_idx = np.random.permutation(X.shape[0])[:sep]
    unlab_idx = np.setdiff1d(np.arange(X.shape[0]), labeled_idx)

    algo_kwargs = vars(algo_parser.parse_known_args()[0])
    algo_kwargs['x_labeled'] = X[labeled_idx]
    algo_kwargs['y_labeled'] = y[labeled_idx]
    algo_kwargs['x_unlabeled'] = X[unlab_idx]
    algo_kwargs['y_unlabeled'] = y[unlab_idx]

    model_kwargs = vars(model_parser.parse_known_args()[0])
    model_kwargs['input_shape'] = X[0].shape
    model = models[I.model_name](**model_kwargs)
    model.fit(X[labeled_idx], y[labeled_idx],
              epochs=model_kwargs['epochs'],
              batch_size=model_kwargs['batch_size'],
              validation_split=.2,
              callbacks=[EarlyStopping(patience=model_kwargs['patience'])])

    algo_kwargs['model'] = model
    algo_kwargs['callbacks'] = [EarlyStopping(patience=model_kwargs['patience'])]
    algo = algodict[I.algorithm](id_string=I.dataset, info_kwargs=vars(I), **algo_kwargs)

    algo.run()


if __name__ == '__main__':
    # let's clean up mess every time we start up some experiment
    from shutil import rmtree
    import os
    from glob import glob
    from paths import HISTORIES
    past_entries = glob('/'.join([HISTORIES, '*']))
    for ent in past_entries:
        if len(os.listdir(ent)) < 2: # at most params.pkl
            rmtree(ent)

    main()