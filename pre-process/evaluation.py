import sys
from os import listdir


def calculate_metrics(datapoints):
    '''
    expects list of tuples as input
        input datapoints = [(true_label, predicted_label)]

    '''

    cm = generate_confusion_matrix(datapoints=datapoints, show=False)

    for i in range(len(cm[0])):
        tp, fn, fp, tn = get_cm_for_class(cm, i)

        if tp + fn > 0:
            recall = max(tp / (tp + fn), 0)
        else:
            recall = 0

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if fp + tn > 0:
            specificity = tn / (fp + tn)
        else:
            specificity = 0

        if precision + recall > 0:
            f1 = 2 * ((precision * recall) / (precision + recall))
            f05 = calculate_f_beta(precision, recall, 0.5)
            f2 = calculate_f_beta(precision, recall, 2)
        else:
            f1, f05, f2 = 0, 0, 0

    accuracy = calculate_accuracy(cm)

    return precision, recall, f1, accuracy


def generate_confusion_matrix(datapoints, show=False):
    '''
    Generates confusion matrix. Class label -1 is equal to index 0
    '''

    cm = [[0 for i in range(3)] for j in range(3)]

    for dp in datapoints:
        true_label = dp[0]
        pred_label = dp[1]

        cm[true_label + 1][pred_label + 1] += 1

    if show:
        print '       predicted'
        print 'true -1 0 1'
        for i in range(3):
            print '  ', i-1, cm[i][0], cm[i][1], cm[i][2]

    return cm


def get_cm_for_class(cm, class_index):
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    tp = cm[class_index][class_index]

    for i in range(len(cm[0])):
        for j in range(len(cm[i])):
            if j == class_index:
                if i != class_index:
                    fn += cm[i][j]

            if i == class_index:
                if j != class_index:
                    fp += cm[i][j]

            if i != class_index and j != class_index:
                tn += cm[i][j]

    return tp, fn, fp, tn


def calculate_accuracy(cm):
    total = 0
    tp = 0

    for i in range(len(cm[0])):
        for j in range(len(cm[i])):
            total += cm[i][j]
            if i == j:
                tp += cm[i][j]

    return float(tp / total)


def calculate_f_beta(precision, recall, beta):
    return (1 + pow(beta, 2)) * ((precision * recall) / ((pow(beta, 2) * precision) + recall))


def parse_files(file1, file2):
    true_labels = [int(line.split()[0]) for line in open(file1, 'r').readlines()]
    pred_labels = [int(line) for line in open(file2, 'r').readlines()]

    datapoints = []
    for i in range(len(true_labels)):
        datapoints.append((true_labels[i], pred_labels[i]))

    return datapoints


def find_labels_file(path, runid):
    # print path, run_id
    for fi in listdir(path):
        # print path, fi, runid
        if runid in fi:
            return "{}/{}".format(path, fi)

def run_evaluation(predictions_root, labels_root):
    topics = ['t10', 't15', 't20', 't25', 't30', 't35', 't40', 't45', 't50']

    # labels_root = "/home/yiren/Documents/Financial-Topic-Model/data/features/cross_validation/test.true.labels/"
    # predictions_root = "/home/yiren/Documents/Financial-Topic-Model/codes/classifier/lr/results/l2.c1.0/"

    for topic in topics:
        print '---------------------------'
        num_topic = topic.replace('t', '')
        lsdirs = sorted(listdir(predictions_root))
        for folder in lsdirs:
            if topic not in folder:
                continue
            features_name = folder.replace("."+topic, "")
            # print folder

            precision = 0.
            recall = 0.
            f1 = 0.
            accuracy = 0.
            count = 0

            for lsdir in listdir(predictions_root + folder):
                if 'output' in lsdir:
                    run_id = lsdir.replace('output.{}.'.format(topic), '').replace('.txt', '')
                    s = run_id.split(".")[0].replace('s', '')
                    k = run_id.split(".")[1].replace('k', '')
                    # svm output file
                    pred_labels_file = "{}{}/{}".format(predictions_root, folder, lsdir)
                    # test set file
                    features_folder_name = "topic_dist_{0}{1}".format(topic.replace('t', ''),
                                                                      features_name.replace('topic', ''))
                    if features_name == "topic_dist":
                        features_folder_name = "topic_dist_{0}".format(topic.replace('t', ''))
                    true_labels_file = "{0}{1}/{2}/{3}/{2}.test.s{3}.k{4}".format(labels_root,
                                                                                  features_name,
                                                                                  features_folder_name,
                                                                                  s, k)
                    # true_labels_file = find_labels_file(true_labels_path, run_id)
                    #print "true:", true_labels_file
                    #print "test:", pred_labels_file
                    p, r, f, a = calculate_metrics(parse_files(true_labels_file, pred_labels_file))
                    precision += p
                    recall += r
                    f1 += f
                    accuracy += a
                    count += 1
            precision = precision / count * 100
            recall = recall / count * 100
            f1 = f1 / count * 100
            accuracy = accuracy / count * 100
            print folder, 'precision=', "{0:.2f}".format(precision), 'recall=', "{0:.2f}".format(recall), \
                  'f1=', "{0:.2f}".format(f1), 'accuracy=', "{0:.2f}".format(accuracy)


if __name__ == '__main__':
    #st = "hist.kernel2.c1024"
    #st = "hist_cont.kernel2.c1024"
    #st = "hist.sentiment.kernel2.c1024"
    st = "hist_cont.sentiment.kernel2.c1024"
    predictions_root = "/home/yiren/Documents/Financial-Topic-Model/codes/classifier/svm/results/{}/".format(st)
    labels_root = "/home/yiren/Documents/Financial-Topic-Model/data/features/cross_validation/"
    run_evaluation(predictions_root, labels_root)
