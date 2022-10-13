import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

# https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/4?u=noobcoder
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, batch_size):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: list(np.where(np.array(self.labels)==label)[0])
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}

        self.batch_size = batch_size
        self.n_classes  = len(self.labels_set)
        divide = list(map(len, np.array_split(range(self.batch_size),
                                                    self.n_classes)))
        divide = [num for _, num in sorted(zip(list(map(len,self.label_to_indices.values())), divide), reverse=True)]
        self.n_samples = {label: divide[idx] for idx, label in enumerate(self.labels_set)}

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.labels):
            indices = []
            for class_ in self.labels_set:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples[class_]])
                self.used_label_indices_count[class_] += self.n_samples[class_]
                if self.used_label_indices_count[class_] + self.n_samples[class_] > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices
            self.count += self.batch_size

    def __len__(self):
        return len(self.labels) // self.batch_size
