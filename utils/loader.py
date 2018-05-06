from os.path import join
from os import listdir
import io
from math import floor

DATA_DIR = 'data'
TRAIN_LABELS = join(DATA_DIR, 'train.tsv')
DEV_LABELS = join(DATA_DIR, 'dev.tsv')

LABELED_DIR = join(DATA_DIR, 'labeled')
UNLABELED_DIR = join(DATA_DIR, 'unlabeled')

STOPWORDS = join(DATA_DIR, 'nltk_stopwords.txt')

def load_labels(file):
    with io.open(file, encoding='utf-8') as fd:
        # Split on line breaks, and remove whitespace
        split_contents = [l.strip() for l in fd.read().split('\n')]
        # Split lines into {text file: candidate} mappings
        return {split.split('\t')[0]: split.split('\t')[1] for split in split_contents if len(split) > 0}

def get_data(label_loc, data_loc):
    print('get_data')
    candidate_samples = {}
    for file_name, candidate in load_labels(label_loc).items():
        if candidate not in candidate_samples:
            candidate_samples[candidate] = []
        with io.open(join(data_loc, file_name), encoding='utf-8') as fd:
            candidate_samples[candidate] += [fd.read()]
    return candidate_samples

def get_labeled_train():
    return get_data(TRAIN_LABELS, DATA_DIR)

def get_dev():
    return get_data(DEV_LABELS, DATA_DIR)

def get_unlabeled_train(percentage=1.0):
    percentage = min(1.0, percentage)
    unlabeled = []
    unlabeled_files = listdir(UNLABELED_DIR)
    for f in unlabeled_files[:floor(percentage*len(unlabeled_files))]:
        with io.open(join(UNLABELED_DIR, f), encoding='utf-8') as fd:
            unlabeled += [fd.read()]
    return unlabeled

def get_stopwords():
    with io.open(STOPWORDS, encoding='utf-8') as fd:
        return [w.strip() for w in fd.read().split('\n') if len(w.strip()) > 0]