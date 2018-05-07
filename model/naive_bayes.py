import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utils import transform
from math import log
from functools import reduce

class NaiveBayesClassifier(object):

    UNKNOWN = ''
    SMOOTHING = 0.0001

    def __init__(self):
        self.p_label = {}
        self.data = {}
        self.labeled_samples = {}

    """Trains the model, takes data in the form of {label -> [samples]}"""
    def train_labeled(self, data):
        self.labeled_samples = data
        self.data = {label: transform.to_bag_of_words(samples) for label, samples in data.items()}

        # Add unknown - represented by the empty string
        for label in data:
            self.data[label][self.UNKNOWN] = self.SMOOTHING

        # Determine p(label)
        self.p_label = {label: len(self.labeled_samples[label]) for label in self.labeled_samples}

        # Re-normalize
        num_samples = 0
        for label, samples in self.labeled_samples.items():
            num_samples += len(samples)

        self.p_label = {label: count/num_samples for label, count in self.p_label.items()}

        all_words = reduce(lambda x, y: x | y.keys(), self.data.values(), set())
        for word in all_words:
            occurrences = sum(word in self.data[label] for label in self.data)
            weight = log(1 + len(self.data)/occurrences)
            for label in self.data:
                if word in self.data[label]:
                    self.data[label][word] *= weight


    """Trains the model, takes data in the form of [samples]"""
    def train_unlabeled(self, data, iterations=5):
        og_labeled = dict(self.labeled_samples)
        for it in range(iterations):
            updated = dict(og_labeled)
            max_logs = {}
            for sample in data:
                max_log, label = self.classify(sample)
                max_logs[max_log] = sample
                # Update only those we were pretty sure about
                if max_log > -50:
                    updated[label].append(sample)
            self.train_labeled(updated)

    def classify(self, observed):
        p_label_given_data = {}
        # Start with non-conditional probabilities
        for label in self.data:
            p_label_given_data[label] = log(self.p_label[label])

        # Iterate through words, factoring their occurrence into the probability of a category
        for word, count in transform.to_bag_of_words([observed]).items():
            for label in self.data:
                smoothed_data_count = self.data[label].get(word, self.data[label][self.UNKNOWN])
                p_label_given_data[label] += count * log(smoothed_data_count/len(self.labeled_samples[label]))

        return max(p_label_given_data.values()), max(p_label_given_data, key=lambda k: p_label_given_data[k])

    """Resets the history of the model"""
    def reset(self):
        self.data = None
