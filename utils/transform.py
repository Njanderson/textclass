from .loader import get_stopwords
from re import sub

# Converts samples, a list of sample sentences, to a bag of words model
def to_bag_of_words(samples, remove_stopwords=False):
    counts = {}
    for sample in samples:
        split_stripped_lowered = [sub('[.,\\\\]', '', split.strip().lower()) for split in sample.split(' ')]
        stopwords = get_stopwords() if remove_stopwords else []
        filtered = [f for f in split_stripped_lowered if len(f) > 0 and f not in stopwords]
        for word in filtered:
            counts[word] = counts.get(word, 0) + 1
    return counts

