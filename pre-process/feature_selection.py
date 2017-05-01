import cPickle as pkl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import  SelectKBest, chi2

class DataPoints:
    def __init__(self):
        self.x = []
        self.y = []

    def set(self, data):
        self.x = data[0]
        self.y = data[1]

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def clear(self):
        self.x = []
        self.y = []

class DataReader:
    """
    reader for logistic regression input
    data format: train, test, valid set
                 each set contains one list of documents (each doc as a list of words),
                 and one corresponding labels (integer 1/-1)
    """
    def __init__(self, dataset):
        self.train = DataPoints()
        self.valid = DataPoints()
        self.test = DataPoints()
        self.load_data(dataset)

    def load_data(self, dataset_path):
        print "loading data ...",
        with open(dataset_path, "rb") as f:
            self.train.set(pkl.load(f))
            self.test.set(pkl.load(f))
            self.valid.set(pkl.load(f))
        print "done! train:", len(self.train.x),\
              "valid:", len(self.valid.x), "test:", len(self.test.x)


def get_ngram_chi_scores(data_reader, scores_out, max_order, topK=None):
    print "calculating chi-square scores for {}-grams".format(max_order)

    vocab = None
    vocab_size = None
    scores = []

    count_vectorizer= CountVectorizer(vocabulary=vocab,
                                      max_features=vocab_size,
                                      ngram_range=(1, max_order))
    x_train = count_vectorizer.fit_transform(data_reader.train.x)
    vocab = count_vectorizer.vocabulary_
    ch2 = SelectKBest(chi2, "all")
    ch2_train = ch2.fit_transform(x_train, data_reader.train.y)
    feature_names = count_vectorizer.get_feature_names()
    for i, score in enumerate(ch2.scores_):
        scores.append((score, feature_names[i]))

    scores.sort(reverse=True)

    with open(scores_out, 'w') as out:
        if topK is None:
            [out.write('{},{}\n'.format(score[1], score[0])) for score in scores]
        else:
            [out.write('{},{}\n'.format(score[1], score[0])) for score in scores[:topK]]


if __name__ == '__main__':
    dir_data = "/Users/ds/git/financial-topic-modeling/data/bpcorpus/"
    f_dataset_docs = dir_data + "dataset/corpus_bp_cls.npz"
    unigram_scores_out = dir_data + "chi-unigram-scores.csv"
    bigram_scores_out = dir_data + "chi-bigram-scores.csv"

    data_reader = DataReader(dataset=f_dataset_docs)
    get_ngram_chi_scores(data_reader, unigram_scores_out, 1)
    get_ngram_chi_scores(data_reader, bigram_scores_out, 2, topK=57216)
