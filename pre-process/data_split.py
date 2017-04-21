import random
import numpy as np
from datetime import date
from data_separate import to_date

def separate_train_test_validation(split_date, perc_val, f_label, f_split, balance=False):
    '''
    Format:
    Company, Date, Id, Open, Close, Label, Dataset

    :param split_date: Date for splitting training and testing
    :param perc_val: Percentage value (< 1) of testing data that will be used as validation set
    :param f_label: Path to croups label file
    :param f_split: Output
    '''

    print "Splitting dataset on date: {}...".format(str(split_date))

    with open(f_label, "r") as f:
        lines = f.readlines()

    fout = open(f_split, "w")

    train = []
    test = []

    for line in lines:
        if to_date(line.split(',')[1]) <= split_date:
            train.append(line)
        else:
            test.append(line)

    validation = []

    random.shuffle(train)

    for i in range(int(len(train) * perc_val)):
        validation.append(train.pop())

    if balance:
        train = balance_dataset_binary(train)
        test = balance_dataset_binary(test)
        if len(validation) > 0:
            validation = balance_dataset_binary(validation)

    num_train = len(train)
    num_test = len(test)
    num_val = len(validation)

    for line in train:
        fout.write("{},{}\n".format(line.strip(), 0))

    for line in test:
        fout.write("{},{}\n".format(line.strip(), 1))

    for line in validation:
        fout.write("{},{}\n".format(line.strip(), 2))

    total = 0.0 + num_val + num_test + num_train

    print "done! {}% training instances, {}% test instances, {}% validation instances." \
        .format(num_train / total, num_test / total, num_val / total)


def split_by_label(dataset):
    label_split = dict()

    for line in dataset:
        label = int(line.strip().split(',')[-1])
        if label not in label_split:
            label_split[label] = []
        label_split[label].append(line)

    return label_split


def balance_dataset_binary(dataset):
    label_split = split_by_label(dataset)
    if len(label_split[0]) > len(label_split[1]):
        label_split[1].extend(np.random.choice(label_split[1], len(label_split[0]) - len(label_split[1])))
    else:
        label_split[0].extend(np.random.choice(label_split[0], len(label_split[1]) - len(label_split[0])))

    print len(label_split[0]), len(label_split[1])

    label_split[0].extend(label_split[1])
    return label_split[0]



if __name__ == '__main__':
    data_path = "/Users/ds/git/financial-topic-modeling/data/bpcorpus/"
    f_labels = data_path + "corpus_labels.csv"
    f_split = data_path + "corpus_labels_split.csv"
    split_date = date(2012, 6, 7)

    separate_train_test_validation(split_date, 0.0, f_labels, f_split, balance=True)
