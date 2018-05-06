from .loader import get_stopwords
# Converts samples, a list of sample sentences, to a bag of words model
def to_bag_of_words(samples, remove_stopwords=True):
    counts = {}
    for sample in samples:
        split_stripped = [split.strip() for split in sample.split(' ')]
        stopwords = get_stopwords()
        filtered = [f for f in split_stripped if len(f) > 0 and f not in stopwords]
        for word in filtered:
            counts[word] = counts.get(word, 0) + 1
    return counts

