from .transform import *

def evaluate_model(model, dev):
    correct = {}
    total = {}
    errors = {}
    for label, samples in dev.items():
        errors[label] = {}
        total[label] = len(samples)
        correct[label] = 0
        for sample in samples:
            p, predicted_label = model.classify(sample)
            if predicted_label == label:
                correct[label] += 1
            else:
                errors[label][predicted_label] = errors[label].get(predicted_label, 0) + 1
                # print('Wanted label %s, but found %s' % (label, predicted_label))
    return errors, {label : correct[label]/total[label] for label in total}
