import numpy as np
import os
import random
from datetime import date

class DataPoint(object):

    def __init__(self, label, values, index):
        self.label = label
        self.values = values
        self.index = index


def to_date(string):
    split = string.split('-')
    return date(int(split[0]), int(split[1]), int(split[2]))


def read_dataset(path):
    '''
    :param path:  Path to the dataset
    :return: A list of DataPoint
    '''
    f = open(path, 'r')

    dataset = []

    for i, line in enumerate(f.readlines()):
        line = line.replace('\n', '')
        splits = line.split()
        label = float(splits[0])
        values = dict()
        for split in splits[1:]:
            feature = int(split.split(':')[0])
            value = float(split.split(':')[1])
            values[feature] = value
        dataset.append(DataPoint(label, values, i))

    return dataset


def sample_balanced_dataset(dataset, shuffle=True, sample=True):
    '''
    Sample data set that has as many positive instances
     as it has negative instances
    '''

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
    '''
    Returns k datasets for train and test, corresponding to
    k folds each having the same amount of negative and positive samples
    '''


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


def oversample(dataset):
    pos = []
    neg = []
    neu = []

    for datapoint in dataset:
        if datapoint.label == 1:
            pos.append(datapoint)
        elif datapoint.label == -1:
            neg.append(datapoint)
        elif datapoint.label == 0:
            neu.append(datapoint)
        else:
            print "ERROR: No label for datapoint: ", datapoint

    while len(pos) < len(neu):
        pos.append(pos[random.randint(0, len(pos)-1)])

    while len(neg) < len(neu):
        neg.append(neg[random.randint(0, len(neg)-1)])

    # add all datapoints together into 'pos'
    pos.extend(neg)
    pos.extend(neu)

    return pos


def generate_dataset_split(batch_name, features_root, corpus_split, oversampling=False):
    path_data = features_root + batch_name + '/'
    path_time = features_root + 'time/'
    path_obj = path_time + batch_name + '/'


    try:
        os.stat(path_time)
    except:
        os.mkdir(path_time)

    try:
        os.stat(path_obj)
    except:
        os.mkdir(path_obj)

    with open(corpus_split, "r") as f:
        split_info = f.readlines()



    for data_file in os.listdir(path_data):
        print "generation training data for", data_file
        file_name = data_file.replace(".csv", "")
        file_name = file_name.replace(".txt", "")
        path_out = path_obj + file_name + "/"

        try:
            os.stat(path_out)
        except:
            os.mkdir(path_out)


        data = read_dataset(path_data + data_file)

        train = []
        test = []
        valid = []

        for datapoint in data:
            dataset = int(split_info[datapoint.index].strip().split(',')[6])
            if dataset == 0:
                train.append(datapoint)
            elif dataset == 1:
                test.append(datapoint)
            elif dataset == 2:
                valid.append(datapoint)
            else:
                print "ERROR: No dataset could be found for datapoint: {}".format(str(datapoint))

        if oversampling:
            train = oversample(train)
            test = oversample(test)
            valid = oversample(valid)

        f_valid = path_out + file_name + ".valid"
        try:
            os.stat(f_valid)
        except:
            valid_to_file(valid, f_valid)

        train_test_to_file(train=train, test=test,
                           train_out=path_out + file_name + '.train',
                           test_out=path_out + file_name + '.test')



if __name__ == '__main__':

    ###### Five-fold cross-validation #####
    '''
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

    '''

    ###### Time based training and testing ######
    '''
    path = '../data/features/'
    params_decay = [0.9, 0.7]
    params_window_size = [1, 2]
    split_date = date(2015,6,1)

    for decay in params_decay:
        for window_size in params_window_size:
            st = "topic_hist_d{}_w{}".format(decay, window_size)
            generate_dataset_time(st=st, path=path, split_date=split_date)
            st = "topic_hist_d{}_w{}_cont".format(decay, window_size)
            generate_dataset_time(st=st, path=path, split_date=split_date)
    '''

    ###### Use train, test, val split from corpus_split.csv #####

    path = '../data/features/'
    corpus_split = '../data/lda/corpus_split_regression.csv'
    oversampling = False
    params_decay = [1, 0.9, 0.8, 0.7]
    params_window_size = [1, 2, 3, 4, 5, 6]

    st = "topic_change"
    generate_dataset_split(batch_name=st, features_root=path, corpus_split=corpus_split, oversampling=oversampling)

    for decay in params_decay:
        for window_size in params_window_size:
            st = "topic_hist_d{}_w{}".format(decay, window_size)
            generate_dataset_split(batch_name=st, features_root=path, corpus_split=corpus_split, oversampling=oversampling)
            st = "topic_hist_d{}_w{}_cont".format(decay, window_size)
            generate_dataset_split(batch_name=st, features_root=path, corpus_split=corpus_split, oversampling=oversampling)



