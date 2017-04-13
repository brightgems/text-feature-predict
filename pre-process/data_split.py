import random
from datetime import date
from data_separate import to_date

def separate_train_test_validation(split_date, perc_val, f_label, f_split):
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

    num_train = 0
    num_test = 0
    num_val = 0

    for line in lines:
        data_set_id = -1

        if line in train:
            data_set_id = 0
            num_train += 1
        elif line in test:
            data_set_id = 1
            num_test += 1
        elif line in validation:
            data_set_id = 2
            num_val += 1

        fout.write(line.strip() + ",{}\n".format(str(data_set_id)))

    total = 0.0 + num_val + num_test + num_train

    print "done! {}% training instances, {}% test instances, {}% validation instances." \
        .format(num_train / total, num_test / total, num_val / total)


if __name__ == '__main__':
    data_path = "/Users/ds/git/financial-topic-modeling/data/bpcorpus/"
    f_labels = data_path + "corpus_labels.csv"
    f_split = data_path + "corpus_lablels_split.csv"
    split_date = date(2012, 5, 15)

    separate_train_test_validation(split_date, 0.0, f_labels, f_split)
