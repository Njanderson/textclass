from utils.loader import *
from utils.transform import to_bag_of_words
from model.naive_bayes import NaiveBayesClassifier
from utils.evaluate import evaluate_model
import pprint

# TODO: Add all this stuff
# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument("-m", "--model", dest="model",
#                     help="Model to use", required=True)
# parser.add_argument("-d", "--data", dest="data",
#                     help="The directory corresponding to training data")
# parser.add_argument("-s", "--seed", dest="seed",
#                     help="The random seed to be used")

def evaluate_classifier(classifier):
    dev = get_dev()
    errors, eval = evaluate_model(classifier, dev)
    total = 0
    num_right = 0
    for label in sorted(eval, key=lambda k: eval[k]):
        label_total = len(dev[label])
        label_num_right = eval[label] * label_total
        print('For label=%s, %d/%d were correctly classified. Accuracy=%f' % (
        label, label_num_right, label_total, eval[label]))
        total += label_total
        num_right += label_num_right
    print('Overall: %d/%d=%f' % (num_right, total, num_right / total))

if __name__ == '__main__':
    classifier = NaiveBayesClassifier()
    classifier.train_labeled(get_labeled_train())
    evaluate_classifier(classifier)
    classifier.train_unlabeled(get_unlabeled_train(0.1))
    evaluate_classifier(classifier)


