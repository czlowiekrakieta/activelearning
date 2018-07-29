from abc import abstractmethod
import numpy as np
import pickle
from random import sample
from paths import HISTORIES
from datetime import datetime
from os.path import join
from os import mkdir

# TODO: history of chosen samples
# TODO: regression implementation


def compute_uncertainty(proba_preds, method):
    order = np.argsort(proba_preds, axis=1)
    N = proba_preds.shape[0]
    # might be unnecessary overkill, but probably won't be bottleneck
    if method == 'ms':
        value = proba_preds[range(N), order[:, -1]] - proba_preds[range(N), order[:, -2]]
    elif method == 'lc':
        value = proba_preds[range(N), order[:, -1]]
    else:
        value = np.sum(proba_preds * np.log(proba_preds), axis=1)

    return value


def select_best_samples(candidates, values, final_batch_size, method='most_uncertain'):
    if method not in {'most_uncertain', 'most_diverse'}:
        raise NameError

    """
    values - the lower, the more we would like it to make it labeled
    for margin sampling it means that difference between best and second best is low
    for least confidence - probability of highest class is low
    for entropy - lowest means that its absolute value is highest
    
    """
    if method == 'most_uncertain':
        order = np.argsort(values)
        return order[:final_batch_size]
    else:
        raise NotImplementedError


class ActiveLearningTask:
    """
    Base abstract class.

    """
    def __init__(self, model, x_labeled, y_labels, x_unlabeled,
                 y_unlabeled, task, candidate_batch=320, final_batch=32, update_every_k_iterations=5,
                 variational_dropout_copies=10, method='en', dump_history_every_k_iterations=30,
                 iterations=1000, id_string=None, info_kwargs=None, *args, **kwargs):
        if task.lower() not in {'regression', 'classification'}:
            raise NameError("task variable must be a string equal to "
                            "either 'regression' or 'classification'")

        self.vd_copies = variational_dropout_copies
        self.model = model
        self.XL = x_labeled
        self.YL = y_labels
        self.XU = x_unlabeled
        self.YU = y_unlabeled
        self.task = task
        self.candidate_batch = candidate_batch
        self.final_batch = final_batch
        self.fit_kwargs = kwargs
        self.ueki = update_every_k_iterations
        self.rounds_since_update = 0
        self.available_xu_idx = np.arange(self.XU.shape[0])
        self.dheki = dump_history_every_k_iterations
        self.ID = id_string

        self.date = datetime.now().strftime('%c')

        if task == 'classification' and method.lower() not in {'ms', 'lc', 'en'}:
            raise NameError("method variable has to be string equal to ms, lc, en. Respectively: "
                            "margin sampling, least confidence, entropy.")
        self.method = method

        self.queried_samples_idx = []
        self.queried_labels = []

        self.history = []

        self.iters = iterations
        self.IK = info_kwargs or {}

        self.path = join(HISTORIES, self.experiment_id)
        mkdir(join(HISTORIES, self.experiment_id))

    def run(self):
        with open(join(self.path, 'params.pkl'), 'wb') as f:
            pickle.dump(self.params(), f)

        for i in range(self.iters):
            self.one_round()

            if i % self.dheki:
                with open(join(self.path, 'HISTORY-' + str(i) + '.pkl'), 'wb') as f:
                    pickle.dump(self.history, f)

                self.history = []

            print('ITERATIONS', i)

    @abstractmethod
    def one_round(self):
        raise NotImplementedError("ActiveLearning class is not supposed to be used directly. "
                                  "Use one of its derived classes.")

    @abstractmethod
    def params(self):
        raise NotImplementedError("ActiveLearning class is not supposed to be used directly. "
                                  "Use one of its derived classes.")

    def base_params(self):
        p = {
            'candidate_batch': self.candidate_batch,
            'final_batch': self.final_batch,
            'method': self.method,
            'task': self.task,
        }
        p.update(self.IK)
        return p

    @property
    def experiment_id(self):
        L = [self.NAME, self.task, self.date]
        if self.ID is not None:
            L = [self.ID] + L

        for n in ['model_name', 'dataset']:
            if n in self.IK:
                L.append(self.IK[n])
        return '__'.join(L)


class QueryByCommittee(ActiveLearningTask):
    pass


class UncertaintySampling(ActiveLearningTask):
    """
    Class for uncertainty sampling

    """

    NAME = 'US'

    def params(self):
        return self.base_params()

    def __init__(self, model, x_labeled, y_labeled, x_unlabeled, y_unlabeled, task,
                 candidate_batch=320, final_batch=32, method='ms', iterations=100, *args, **kwargs):

        super(UncertaintySampling, self).__init__(model, x_labeled,
                                                  y_labels=y_labeled,
                                                  x_unlabeled=x_unlabeled,
                                                  y_unlabeled=y_unlabeled,
                                                  iterations=iterations,
                                                  task=task,
                                                  candidate_batch=candidate_batch,
                                                  final_batch=final_batch,
                                                  method=method)

    def one_round(self):
        candidates_idx = np.random.permutation(self.XU.shape[0])[:self.candidate_batch]
        candidates = self.XU[candidates_idx]

        if self.task == 'classification':
            proba_preds = self.model.predict_proba(candidates)
            value = compute_uncertainty(proba_preds, self.method)
            low_confidence_idx = select_best_samples(candidates, values=value, final_batch_size=self.final_batch)
            lc_pool_idx = candidates_idx[low_confidence_idx]

            self.queried_samples_idx.extend(lc_pool_idx)
            self.queried_labels.extend(self.YU[lc_pool_idx])

            h_dict = {
                'queried_idx': lc_pool_idx,
                'queried_labels': self.YU[lc_pool_idx],
                'candidates_values': value,
                'candidates_idx': candidates_idx,
            }

            self.history.append(h_dict)

        else:
            raise NotImplementedError

        return candidates_idx, value


class CEAL(ActiveLearningTask):

    """
    Implementation of Cost-Effective Active Learning for Deep Image Classification

    https://arxiv.org/pdf/1701.03551.pdf
    """

    NAME = 'CEAL'

    def __init__(self, model, x_labeled, y_labeled, x_unlabeled, y_unlabeled, task='classification', method='ms',
                 entropy_threshold=0.05, mean_to_variance_threshold=None, candidate_batch=320, iterations=100,
                 final_batch=32, update_every_k_iterations=5, *args, **kwargs):

        super(CEAL, self).__init__(model, x_labeled, y_labels=y_labeled, x_unlabeled=x_unlabeled, y_unlabeled=y_unlabeled,
                                   task=task, method=method, candidate_batch=candidate_batch, final_batch=final_batch,
                                   update_every_k_iterations=update_every_k_iterations, iterations=iterations)

        self.et = entropy_threshold
        self.mtv = mean_to_variance_threshold

        self.pseudo_labeled_idx = []
        self.pseudo_labels = []

    def params(self):
        p = self.base_params()
        p.update({'entropy_threshold': self.et, 'mean_to_variance': self.mtv})
        return p

    def one_round(self):
        candidates_idx = np.random.permutation(self.available_xu_idx)[:self.candidate_batch]
        candidates = self.XU[candidates_idx]

        if self.task == 'classification':
            proba_preds = self.model.predict_proba(candidates)
            weak_value = compute_uncertainty(proba_preds, self.method)
            if self.method != 'en':
                entropy = - compute_uncertainty(proba_preds, 'en')
            else:
                entropy = - weak_value.copy()

            high_confidence_idx = np.where(entropy < self.et)
            high_confidence_labels = np.argmax(proba_preds, axis=1)[high_confidence_idx]

            low_confidence_idx = select_best_samples(candidates, weak_value, self.final_batch)
            lc_pool_idx = candidates_idx[low_confidence_idx]
            low_confidence_labels = self.YU[lc_pool_idx]

            self.pseudo_labeled_idx.extend(candidates_idx[high_confidence_idx].tolist())
            self.pseudo_labels.extend(high_confidence_labels.tolist())

            self.queried_samples_idx.extend(lc_pool_idx.tolist())
            self.queried_labels.extend(low_confidence_labels)

            self.available_xu_idx = np.setdiff1d(self.available_xu_idx,
                                                 self.queried_samples_idx + self.pseudo_labeled_idx)

        else:
            raise NotImplementedError

        self.rounds_since_update += 1
        if self.rounds_since_update == self.ueki:
            x_update = self.XU[self.pseudo_labeled_idx + self.queried_samples_idx]
            y_update = self.pseudo_labels + self.queried_labels
            self.model.fit(x_update, y_update, **self.fit_kwargs)

            """
            from the paper itself:
            
            After  fine-tuning  we  put  the  high  confidence  samples D_H back to
            D_U and erase their pseudo-label
            """

            self.available_xu_idx = np.union1d(self.available_xu_idx, self.pseudo_labeled_idx)
            self.pseudo_labeled_idx = []
            self.pseudo_labels = []

        self.rounds_since_update %= self.ueki


class Random(ActiveLearningTask):
    """
    Random baseline for comparison

    """

    NAME = 'RANDOM'

    def one_round(self):
        candidates_idx = np.random.permutation(self.available_xu_idx)[:self.final_batch]
        candidates_labels = self.YU[candidates_idx]

        self.queried_samples_idx.extend(candidates_idx)
        self.queried_labels.extend(candidates_labels)

        self.available_xu_idx = np.setdiff1d(self.available_xu_idx, self.queried_samples_idx)

        x_update = self.XU[self.queried_samples_idx]
        self.model.fit(x_update, self.queried_labels, **self.fit_kwargs)