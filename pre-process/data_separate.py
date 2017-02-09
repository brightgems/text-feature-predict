import numpy as np
import os
import random

class DataPoint(object):

    def __init__(self, label, values):
        self.label = label
        self.values = values


def read_dataset(path):
    f = open(path, 'r')

    dataset = []

    for line in f.readlines():
        line = line.replace('\n', '')
        splits = line.split()
        values = dict()
        for split in splits[1:]:
            feature = int(split.split(':')[0])
            value = float(split.split(':')[1])
            values[feature] = value
        dataset.append(DataPoint(int(splits[0]), values))

    return dataset


def sample_balanced_dataset(dataset, shuffle=True, sample=True):
    positive = []
    negative = []

    for datapoint in dataset:
        if datapoint.label == 1:
            positive.append(datapoint)
        elif datapoint.label == 0:
            negative.append(datapoint)
        else:
            print('datapoint label incorrect')
            exit(1)

    if shuffle:
        random.shuffle(positive)
        random.shuffle(negative)

    if sample:
        return positive, np.random.choice(negative, len(positive), True)
    else:
        return positive, negative


def k_fold_datasets(positive, negative, k):
    train = [0] * k
    test = [0] * k

    folds_pos = np.array_split(positive, k)
    folds_neg = np.array_split(negative, k)

    for i in range(k):
        test[i] = folds_pos[i].tolist()
        test[i].extend(folds_neg[i].tolist())
        train[i] = list()
        for j in range(k):
            if i != j:
                train[i].extend(folds_pos[j].tolist())
                train[i].extend(folds_neg[j].tolist())

    return train, test

def train_test_to_file(train, test, train_out, test_out):
    tro = open(train_out, 'w')
    teo = open(test_out, 'w')

    for datapoint in train:
        tro.write('{} '.format(datapoint.label))
        for feature, value in datapoint.values.items():
            tro.write('{}:{} '.format(feature, value))
        tro.write('\n')
    tro.close()

    for datapoint in test:
        teo.write('{} '.format(datapoint.label))
        for feature, value in datapoint.values.items():
            teo.write('{}:{} '.format(feature, value))
        teo.write('\n')
    teo.close()

def valid_to_file(valid, valid_out):
    tvo = open(valid_out, "w")
    for datapoint in valid:
        tvo.write('{} '.format(datapoint.label))
        for feature, value in datapoint.values.items():
            tvo.write('{}:{} '.format(feature, value))
        tvo.write('\n')
    tvo.close()

def generate_validation(positive, negative, k=6):
    folds_pos = np.array_split(positive, k)
    folds_neg = np.array_split(negative, k)
    valid = []
    other = []

    valid = folds_pos[k-1].tolist()
    valid.extend(folds_neg[k-1].tolist())

    for i in range(k-1):
        other.extend(folds_pos[i].tolist())
        other.extend(folds_neg[i].tolist())

    return valid, other


def generate_dataset(num_folds, st, path):
    path_data = path + st + '/'
    path_obj = path + 'cross_validation/' + st + '/'
    try:
        os.stat(path_obj)
    except:
        os.mkdir(path_obj)

    for data_file in os.listdir(path_data):
        print "generation training data for", data_file
        file_name = data_file.replace(".csv", "")
        file_name = file_name.replace(".txt", "")
        path_out = path_obj + file_name + "/"
        try:
            os.stat(path_out)
        except:
            os.mkdir(path_out)
        f_valid = path_out + file_name + ".valid"
        f_other = path_out + file_name + ".data"

        # create validation set if not exist
        try:
            os.stat(f_valid)
            os.stat(f_other)
        except:
            data = read_dataset(path_data + data_file)
            pos, neg = sample_balanced_dataset(data, sample=False)
            valid, other = generate_validation(positive=pos, negative=neg, k=6)
            valid_to_file(valid, f_valid)
            valid_to_file(other, f_other)

        # do cross-validation on other (validation excluded)
        # repeat num_folds times
        for k in range(num_folds):
            path_cross = path_out + str(k) + "/"
            try:
                os.stat(path_cross)
            except:
                os.mkdir(path_cross)
            data = read_dataset(f_other)
            pos, neg = sample_balanced_dataset(data, sample=True)  # balanced sampling
            folds_tr, folds_te = k_fold_datasets(pos, neg, num_folds)

            for i in range(num_folds):
                train_test_to_file(folds_tr[i], folds_te[i],
                                   path_cross + file_name + '.train.s{}.k{}'.format(k, i),
                                   path_cross + file_name + '.test.s{}.k{}'.format(k, i))

if __name__ == '__main__':

    num_folds = 5 
    # st = 'topic_combined_sentiment'
    path = '../data/features/'
    sentiment = False
    params_decay = [0.9, 0.7]
    params_window_size = [1, 2]

    st_sen = ""
    if sentiment:
        st_sen = "_sentiment"

    for decay in params_decay:
        for window_size in params_window_size:
            st = "topic_hist_d{}_w{}{}".format(decay, window_size, st_sen)
            generate_dataset(num_folds=num_folds, st=st, path=path)
            st = "topic_hist_d{}_w{}_cont{}".format(decay, window_size, st_sen)
            generate_dataset(num_folds=num_folds, st=st, path=path)