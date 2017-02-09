import sys
from os import listdir

def calculate_metrics(datapoints):
    '''
    expects list of tuples as input
        input datapoints = [(true_label, predicted_label)]
        :returns precision, recall, f1, accuracy
    '''

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for dp in datapoints:
        if dp[0] == 1:
            if dp[0] == dp[1]:
                tp += 1
            elif dp[1] == 0:
                fn += 1
            else:
                print 'error', dp
        elif dp[0] == 0:
            if dp[0] == dp[1]:
                tn += 1
            elif dp[1] == 1:
                fp += 1
        else:
            print 'error', dp

    # print fp, tp, fn, tn
    if tp + fp == 0:
        precision = 0.
    else:
        precision = tp / float(tp + fp)
    if tp + fn == 0:
        recall = 0.
    else:
        recall = tp / float(tp + fn)
    if precision + recall == 0.:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / float(precision + recall)
    accuracy = (tp + tn) / float(len(datapoints))

    return precision, recall, f1, accuracy


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