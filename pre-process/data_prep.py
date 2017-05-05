"""
prepare crawled data for feature extraction
take both json and text file as input
extract date and textual article
"""

import ast # abstract syntax trees
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from operator import itemgetter
import cPickle as pkl

from clean_str import clean_str

def extract_docs(path_in, path_out):
    """
    extract (and merge) all crawled documents in the given path
    :param path_in: path to crawled docs
    :param path_out: path to output extracted docs
    """
    files = os.listdir(path_in)
    try:
        os.stat(path_out)
    except:
        os.mkdir(path_out)

    f_json = defaultdict(list)
    f_text = defaultdict(list)
    companies = []

    for f in files:
        if "json" in f:
            company = f.split("-json")[0].split("_")[-1]
            f_json[company].append(f)
            companies.append(company)
        elif "text" in f:
            company = f.split("-text")[0].split("_")[-1]
            f_text[company].append(f)
        else:
            print "unrecognized file:", f
    companies = list(set(companies))

    for company in companies:
        print "processing", company
        doc_info = []
        doc_contents = []
        # load all the lines with meaningful doc_info
        for f in f_json[company]:
            document_json = open(path_in + f, "r")
            for line in document_json.readlines():
                l = ast.literal_eval(line)['response']['docs']
                if len(l) > 0:
                    for ll in l:
                        ll["pub_date"] = ll["pub_date"].split("T")[0]
                    doc_info.append(l)
        # load all the articles
        for f in f_text[company]:
            document_contents = open(path_in + f, "r")
            for line in document_contents.readlines():
                split = line.split('\t')
                if len(split) == 3:  # TODO: Bug in parsing
                    doc_contents.append((split[0], split[2].replace('\n', '')))

        extract_doc_compary(doc_info=doc_info,
                            doc_contents=doc_contents,
                            f_output=path_out+company+".tsv")

def extract_doc_compary(doc_info, doc_contents, f_output):
    """
    extract date and actual textual articles from raw corpus
    :param doc_info: doc info loaded from json file
    :param doc_contents: actual doc articles from text file
    :param f_output: path to output file, each line: date /t [a list of articles]
    """
    document_output = open(f_output, "w")

    # Create DataFrame that contains all attributes
    doc_attributes = pd.DataFrame()
    for doc in doc_info:
        doc_attributes = doc_attributes.append(pd.DataFrame.from_dict(doc))
    doc_attributes = doc_attributes.reset_index(drop=True)
    doc_attributes = doc_attributes.drop_duplicates(subset='_id') # remove duplicate rows w.r.t. label "_id"

    # Create DataFrame that contains docId and document content
    contents = pd.DataFrame(doc_contents, columns=['_id', 'text'])
    contents = contents.drop_duplicates()

    # Join table of attributes with document content table
    collection = doc_attributes.merge(contents, left_on="_id", right_on="_id")
    # collection[['_id', 'pub_date', 'text']]
    # print collection

    # concatenate documents
    select = collection[['pub_date', 'text']]
    concat = select.groupby('pub_date')
    concat = concat['text'].apply(list)
    # print concat

    # write to output file
    concat.to_csv(document_output, sep='\t', encoding='utf8')

    document_output.close()

class lda_prep:
    def __init__(self, path_in, path_out, vocab_size=50000, stop_words=False, load_collection=False):
        self.path_in = path_in
        self.path_out = path_out
        self.stop_words = stop_words
        self.vocab_size = vocab_size

        self.wordList = defaultdict(int) # word -> freq
        self.vocab = dict()
        self.total_words_cnt = 0.
        self.collections = [] # [company, date, list of document words]

        try:
            os.stat(self.path_out)
        except:
            os.mkdir(self.path_out)

        self.fout = open(self.path_out + "corpus_lda.txt", "w")  # doc-term format input for lda

        if not load_collection:
            self.fin_names = os.listdir(self.path_in)


            for fin_name in self.fin_names:
                self.load_docs(fin_name=fin_name)
        else:
            self.load_collection(path_in)

        print "collection size:", len(self.collections)

        self.prep_vocab()

        #self.prep_doc_term()

    def load_docs(self, fin_name):
        """
        load documents for different companies
        :param fin_name: input file name
               the file is of the same format as output of extract_doc_company()
        """
        print "loading documents from ",
        with open(self.path_in+fin_name, "r") as f:
            lines = f.readlines()
        company_name = fin_name.replace(".tsv", "").split("-")[-1]
        print company_name, "...",

        for line in lines:
            line = line.strip().split("\t") # date, doc_content
            content = clean_str(line[1])
            if len(content) == 0:
                continue
            tokens = nltk.word_tokenize(content.decode('utf-8')) # tokenize
            for token in tokens:
                self.wordList[token] += 1
            self.total_words_cnt += len(tokens)
            self.collections.append([company_name, line[0], tokens])
        print "done! corpus size:", len(lines), "total word count:", self.total_words_cnt

    def load_collection(self, fin_name):
        """
        load documents from a document collection [company_name, date, document]
        the document content in the collection needs to be pre-processed and tokenized
        with a space delimiter
        :param fin_name: the path to the document collection
        """
        print "loading documents from corpus: {}".format(fin_name)
        for line in open(fin_name):
            line = line.strip().split("\t")  # company_name, date, doc_content
            company_name = line[0]
            date = line[1]
            content = line[2]
            if len(content) == 0:
                continue
            tokens = content.split(' ')
            for token in tokens:
                self.wordList[token] += 1
            self.total_words_cnt += len(tokens)
            self.collections.append([company_name, date, tokens])
        print "done! corpus size:", len(self.collections), "total word count:", self.total_words_cnt

    def prep_vocab(self):
        """
        generate vocab (generate index based on word frequency)
        dump vocab to file
        """
        print "generating vocabulary ...",
        self.wordList = self.wordList.items()
        self.wordList = sorted(self.wordList, key=itemgetter(1), reverse=True)

        freq_cov = 0.
        for i in range(min(self.vocab_size, len(self.wordList))):
            self.vocab[self.wordList[i][0]] = i
            freq_cov += self.wordList[i][1]

        with open(self.path_out + "vocab.txt", "w") as f:
            f.writelines(["%s\n" % word[0] for word in self.wordList[:len(self.vocab)]])
        with open(self.path_out + "vocab.pkl", "wb") as f:
            pkl.dump(self.vocab, f)
        print "done!"
        print "unique words:", len(self.wordList)
        print "vocab size:", len(self.vocab)
        print "frequency converage:", float(freq_cov) / self.total_words_cnt

    def comb_docs(self, fout_name='corpus_raw.txt'):
        print "combing docs for different companies ...",
        fout = open(self.path_out+fout_name, "w")
        for doc in self.collections:
            s = doc[0] + "\t" + doc[1] + "\t"
            for word in doc[-1]:
                if word in self.vocab:
                    s += word + " "
            fout.write(s+"\n")
        fout.close()
        print "done!"

    def prep_doc_term(self):
        """
        generate doc-term format file as LDA input
        each line: {#unique_words} {widx:freq}, space separated
        """
        print "preparing doc-term matrix for LDA ...",
        for doc in self.collections:
            content = []
            for word in doc[-1]:
                if word in self.vocab:
                    content.append(self.vocab[word])
            word_freq = defaultdict(int)
            for widx in content:
                word_freq[widx] += 1
            s = str(len(word_freq)) + " "
            for kk, vv in word_freq.iteritems():
                s += str(kk) + ":" + str(vv) + " "
            self.fout.write(s+"\n")
        self.fout.close()
        print "done!"

class DataPoints:
    def __init__(self):
        self.x_doc = []
        self.x_stock = []
        self.x_lda = []
        self.y = []

    def set(self, data):
        self.x_doc = data[0]
        self.y = data[-1]
        if len(data) == 3:
            self.set_stock(data[1])

    def set_doc(self, x_doc):
        self.x_doc = x_doc

    def set_lda(self, x_lda):
        self.x_lda = np.array(x_lda)

    def set_stock(self, x_stock):
        self.x_stock = np.array(x_stock)

    def set_y(self, y):
        self.y = y

    def clear(self):
        self.x_doc = []
        self.x_stock = []
        self.x_lda = []
        self.y = []

class DataProcessor:
    def __init__(self, vocab_size=None, valid_portion=None, overwrite=False, shuffle=True):
        self.train = DataPoints()
        self.valid = DataPoints()
        self.test = DataPoints()
        self.vocab = dict() # vocab for the loaded dataset
        self.vocab_size = vocab_size # take top vocab_size vocab for loaded dataset if not None

        self.train_idx = []  # doc_idx for training
        self.test_idx = []  # doc_idx for test
        self.train_labels = []
        self.test_labels = []
        self.train_stock = [] # each element is a list of stock change (today, prev 20 days)
        self.test_stock = []
        self.train_lda_hist_idx = [] # each element is a list of doc_idx for topic history
        self.test_lda_hist_idx = []
        self.train_lda_change_idx = [] # each element is a list of doc_idx for topic change
        self.test_lda_change_idx = []
        self.train_lda_hist = [] # each element is a historical topic dist (linear comb)
        self.test_lda_hist = []
        self.train_lda_change = [] # each element is a topic change
        self.test_lda_change = []

        self.overwrite = overwrite
        self.use_shuffle = shuffle

        self.sidx_train = [] # shuffled idx list for training set
        self.sidx_test = []

        self.topic_dist = [] # a list of nparray

    def run_docs(self, f_corpus, f_meta_data, f_dataset_out, f_vocab=None, f_sidx=None):
        def _run():
            self.load_metadata(f_meta_data=f_meta_data)
            self.load_corpus(f_corpus=f_corpus)
            if self.use_shuffle:
                self.shuffle(f_sidx=f_sidx)
            self.gen_vocab(f_vocab=f_vocab)
            self.save_data(f_dataset_out=f_dataset_out)

        try:
            os.stat(f_dataset_out)
            print f_dataset_out + " already exist!"
            if self.overwrite:
                print "overwriting ..."
                _run()
        except:
            _run()

    def run_lda(self, dir_lda, f_lda_out, alphas, window_sizes, f_meta_data=None, f_sidx=None):
        """
        output format: a list of topic features
        format: list of [train(ndarray, #sample * #feature), test (ndarray), description]
        """
        print "generating lda features ..."
        if len(self.train_idx) == 0 and f_meta_data:
            self.load_metadata(f_meta_data)
        feature_lda_all = []
        paths = os.listdir(dir_lda)

        def _set_feature(train, test, lda_data, feature=""):
            if self.use_shuffle:
                if len(self.sidx_train) == 0:
                    self.load_sidx(f_sidx)
                print "use shuffling"
                train = [train[idx] for idx in self.sidx_train]
                test = [test[idx] for idx in self.sidx_test]
            lda_data.append([np.array(train), np.array(test), feature])

        for path_lda in paths:
            # for each k
            tmp = self.load_lda(dir_lda + path_lda)
            if tmp == 0:
                continue
            feature_lda = []
            train_lda_today = [self.topic_dist[doc_idx] for doc_idx in self.train_idx]
            test_lda_today = [self.topic_dist[doc_idx] for doc_idx in self.test_idx]
            _set_feature(train_lda_today, test_lda_today, feature_lda, feature="")

            print "generating topic change features"
            self.gen_topic_change()
            feature = "change"
            # _set_feature(self.train_lda_change, self.test_lda_change, feature_lda, feature)

            train_lda = [train_lda_today[i] - self.train_lda_change[i]
                         for i in range(len(train_lda_today))]
            test_lda = [test_lda_today[i] - self.test_lda_change[i]
                        for i in range(len(test_lda_today))]
            _set_feature(train_lda, test_lda, feature_lda, feature)

            print "generating historical features"
            for alpha in alphas:
                print "\talpha={}".format(alpha)
                for window_size in window_sizes:
                    self.gen_topic_hist(alpha=alpha, window_size=window_size)
                    feature = "alpha={}, L={}".format(alpha, window_size)
                    _set_feature(self.train_lda_hist, self.test_lda_hist, feature_lda, feature)

                    """
                    # add together
                    feature = "alpha={}, L={}".format(alpha, window_size)
                    train_lda = [train_lda_today[i] + self.train_lda_hist[i]
                                 for i in range(len(train_lda_today))]
                    test_lda = [test_lda_today[i] + self.test_lda_hist[i]
                                for i in range(len(test_lda_today))]
                    _set_feature(train_lda, test_lda, feature)

                    # concatenate
                    feature += " (cont)"
                    train_lda = [np.concatenate((train_lda_today[i], self.train_lda_hist[i]))
                                 for i in range(len(train_lda_today))]
                    test_lda = [np.concatenate((test_lda_today[i], self.test_lda_hist[i]))
                                 for i in range(len(test_lda_today))]
                    _set_feature(train_lda, test_lda, feature)
                    """

            feature_lda_all.append(feature_lda)


        # save
        print "saving to file...",
        with open(f_lda_out, "wb") as f:
            pkl.dump(feature_lda_all, f)
        print "done!"
        print "\t#different number of topics: {}".format(len(feature_lda_all))
        print "\t#different history combination: {}".format(len(feature_lda_all[0]))

    def reset_idx(self):
        self.train_idx = []  # doc idx for training
        self.test_idx = []  # doc idx for test
        self.train_labels = []
        self.test_labels = []
        self.train_stock = []
        self.test_stock = []
        self.train_lda_hist_idx = []
        self.test_lda_hist_idx = []

    def load_metadata(self, f_meta_data):
        """
        :param f_meta_data: meta data, comma separated
                            {company, date, doc_idx (line number of doc in corpus, starting 0),
                            today's stock change, previous 20 days' stock changes,
                            previous 20 day's doc_idx (-1 if non-exist)
                            label, train(0)/test(1)}
        """
        print "Loading metadata...",
        self.reset_idx()

        with open(f_meta_data, "r") as f:
            meta_data = f.readlines()

        for lidx, meta_line in enumerate(meta_data):
            meta_line = meta_line.strip().split(",")
            assert len(meta_line) == 46, \
                "invalid meta data! line {}, length {}".format(lidx+1, len(meta_line))

            # stock changes
            stock_hist = np.array([float(s) for s in meta_line[3:24]]) # today, 20 previous days
            pos = np.argwhere(np.isnan(stock_hist))
            stock_hist[pos] = 0.

            # lda hist
            lda_hist = [int(s) for s in meta_line[-3:-23:-1]] # 20 previous days

            # topic change
            lda_change = [int(meta_line[-22])]  # yesterday only

            # stock label
            label = int(meta_line[-2])
            if label == 0:
                label = -1

            # train/test label
            train_label = int(meta_line[-1])
            if train_label == 0: # train
                self.train_idx.append(int(meta_line[2]))
                self.train_stock.append(stock_hist)
                self.train_labels.append(label)
                self.train_lda_hist_idx.append(lda_hist)
                self.train_lda_change_idx.append(lda_change)
            elif train_label == 1:
                self.test_idx.append(int(meta_line[2]))
                self.test_stock.append(stock_hist)
                self.test_labels.append(label)
                self.test_lda_hist_idx.append(lda_hist)
                self.test_lda_change_idx.append(lda_change)
            else:
                raise ValueError(
                    "warning: fail to recognize train/test label {0} at line {1}".format(meta_line[1], lidx))

        print "done! {} records loaded!".format(len(meta_data))

    def load_corpus(self, f_corpus):
        """
        load data from corpus and corpus mapping file
        :param f_corpus: corpus {company, date, docs}, tap separated
        """
        print "\tloading from {}".format(f_corpus)
        self.train.clear()
        self.test.clear()

        with open(f_corpus, "r") as f:
            corpus = f.readlines()

        # construct training set
        for i, doc_idx in enumerate(self.train_idx):
            doc = word_tokenize(corpus[int(doc_idx)].strip().split("\t")[-1]) # get doc from corpus
            self.train.x_doc.append(doc)
            self.train.x_stock.append(self.train_stock[i])
            self.train.y.append(self.train_labels[i])

        # construct test set
        for i, doc_idx in enumerate(self.test_idx):
            doc = word_tokenize(corpus[int(doc_idx)].strip().split("\t")[-1])
            self.test.x_doc.append(doc)
            self.test.x_stock.append(self.test_stock[i])
            self.test.y.append(self.test_labels[i])

    def load_lda(self, path_lda):
        self.topic_dist = []
        # get metadata
        alpha = 0.
        try:
            with open(path_lda + "/final.other", "r") as f:
                lines = f.readlines()
        except:
            print "[warning] illegal path ignored: {}".format(path_lda)
            return 0.
        print "loading from {}".format(path_lda)
        for line in lines:
            if "alpha" in line:
                alpha = float(line.strip().split()[-1])
                break

        # get topic distribution
        with open(path_lda + "/final.gamma", "r") as f:
            lines = f.readlines()
        for line in lines:
            probs = line.strip().split()
            probs = [float(prob) - alpha for prob in probs]
            probs_sum = sum(probs)
            probs = [prob / probs_sum for prob in probs]
            self.topic_dist.append(np.array(probs))
        return len(self.topic_dist)

    def gen_topic_change(self):
        self.train_lda_change = []
        self.test_lda_change = []

        for i, lda_change in enumerate(self.train_lda_change_idx):
            doc_idx = lda_change[0]
            change = np.zeros(self.topic_dist[0].shape)
            if doc_idx > 0:
                change = self.topic_dist[lda_change[0]]
            self.train_lda_change.append(change)

        for i, lda_change in enumerate(self.test_lda_change_idx):
            doc_idx = lda_change[0]
            change = np.zeros(self.topic_dist[0].shape)
            if doc_idx > 0:
                change = self.topic_dist[lda_change[0]]
            self.test_lda_change.append(change)

    def gen_topic_hist(self, alpha=1., window_size=1):
        self.train_lda_hist = []
        self.test_lda_hist = []

        for i, lda_hist in enumerate(self.train_lda_hist_idx):
            hist = np.zeros(self.topic_dist[0].shape)
            weight = alpha
            for w in xrange(window_size):
                doc_idx = lda_hist[w]
                if doc_idx > 0: # not -1
                    hist += weight * self.topic_dist[doc_idx]
                weight *= alpha
            self.train_lda_hist.append(hist)

        for i, lda_hist in enumerate(self.test_lda_hist_idx):
            hist = np.zeros(self.topic_dist[0].shape)
            weight = alpha
            for w in xrange(window_size):
                doc_idx = lda_hist[w]
                if doc_idx > 0: # not -1
                    hist += weight * self.topic_dist[doc_idx]
                weight *= alpha
            self.test_lda_hist.append(hist)

    def gen_vocab(self, f_vocab=None):
        """
        generate vocab from loaded dataset
        vocab are saved as dict(): word -> idx
        """
        print "generating vocabulary ..."
        def _get_vocab(dataset, wordlist):
            for line in dataset:
                for word in line:
                    wordlist[word] += 1
            return wordlist

        wordlist = defaultdict(int)
        wordlist = _get_vocab(self.train.x_doc, wordlist)
        wordlist = _get_vocab(self.test.x_doc, wordlist)

        wordlist = sorted(wordlist.items(), key=itemgetter(1), reverse=True)
        freq_cnt = 0.
        total_cnt = 0.
        if not self.vocab_size:
            self.vocab_size = len(wordlist)
        for i, word in enumerate(wordlist):
            if len(self.vocab) < self.vocab_size:
                self.vocab[word[0]] = i
                freq_cnt += word[1]
            total_cnt += word[1]
        print "raw vocab size: {}".format(len(wordlist))
        print "final vocab size: {}".format(len(self.vocab))
        print "freq coverage: {}".format(freq_cnt / total_cnt)

        # save vocab
        if f_vocab:
            pkl.dump(self.vocab, open(f_vocab, "wb"))

    def shuffle(self, f_sidx=None):
        print "using shuffling ... ",
        self.sidx_train = np.random.permutation(len(self.train.y))
        self.sidx_test = np.random.permutation(len(self.test.y))
        if f_sidx:
            print "use random index from file ... ",
            with open(f_sidx, "rb") as f:
                self.sidx_train = pkl.load(f)
                self.sidx_test = pkl.load(f)

        # train
        self.train.x_doc = [self.train.x_doc[idx] for idx in self.sidx_train]
        if len(self.train.x_stock) > 0:
            self.train.x_stock = [self.train.x_stock[idx] for idx in self.sidx_train]
        self.train.y = [self.train.y[idx] for idx in self.sidx_train]

        # test
        self.test.x_doc = [self.test.x_doc[idx] for idx in self.sidx_test]
        if len(self.test.x_stock) > 0:
            self.test.x_stock = [self.test.x_stock[idx] for idx in self.sidx_test]
        self.test.y = [self.test.y[idx] for idx in self.sidx_test]
        print "done!"

    def set_valid(self, valid_portion=0.15):
        """
        set valid_portion of training data into validation set
        """
        n_sample = len(self.train.y)
        sidx = np.random.permutation(n_sample)
        n_train = int(np.round(n_sample * (1 - valid_portion)))
        self.valid.x_doc = [self.train.x_doc[idx] for idx in sidx[n_train:]]
        self.train.x_doc = [self.train.x_doc[idx] for idx in sidx[:n_train]]
        if len(self.train.x_stock) > 0:
            self.valid.x_stock = [self.train.x_stock[idx] for idx in sidx[n_train:]]
            self.train.x_stock = [self.train.x_stock[idx] for idx in sidx[:n_train]]
        self.valid.y = [self.train.y[idx] for idx in sidx[n_train:]]
        self.train.y = [self.train.y[idx] for idx in sidx[:n_train]]

    def save_data(self, f_dataset_out):
        print "saving to file ...",
        if len(self.train.x_stock) > 0:
            train = [[" ".join(line) for line in self.train.x_doc], self.train.x_stock, self.train.y]
            test = [[" ".join(line) for line in self.test.x_doc], self.test.x_stock, self.test.y]
            valid = [[" ".join(line) for line in self.valid.x_doc], self.valid.x_stock, self.valid.y]
        else:
            train = [[" ".join(line) for line in self.train.x_doc], self.train.y]
            test = [[" ".join(line) for line in self.test.x_doc], self.test.y]
            valid = [[" ".join(line) for line in self.valid.x_doc], self.valid.y]
        with open(f_dataset_out, "wb") as f:
            pkl.dump(train, f)
            pkl.dump(test, f)
            pkl.dump(valid, f)
        print "done!"
        print "train:", len(self.train.y), "valid:", len(self.valid.y), "test:", len(self.test.y)

    def save_labels(self, f_labels_out):
        print "saving labels to file ..."
        with open(f_labels_out, "wb") as f:
            pkl.dump(self.train.y, f)
            pkl.dump(self.test.y, f)

    def load_sidx(self, f_sidx):
        with open(f_sidx, "rb") as f:
            self.sidx_train = pkl.load(f)
            self.sidx_test = pkl.load(f)

    def save_sidx(self, f_sidx_out):
        print "saving shuffled indexes to file ..."
        with open(f_sidx_out, "wb") as f:
            pkl.dump(self.sidx_train, f)
            pkl.dump(self.sidx_test, f)


if __name__ == "__main__":
    # dir_data = "/home/yiren/Documents/time-series-predict/data/bp/"
    #dir_data = "/Users/Irene/Documents/financial_topic_model/data/bp/"
    dir_data = "/Users/ds/git/financial-topic-modeling/data/bpcorpus/lda_20170505/"
    f_corpus = dir_data + "standard-query-corpus_pp.tsv"
    f_meta_data = dir_data + "corpus_labels_split_balanced_change.csv"
    f_dataset_out = dir_data + "dataset/corpus_bp_stock_cls.npz"
    f_vocab = dir_data + "dataset/vocab_stock.npz"

    dir_lda = dir_data + "lda_result_20170411/"
    f_lda_out = dir_data + "dataset/lda_change_hist.npz"
    f_labels = dir_data + "dataset/labels.npz"
    f_sidx = dir_data + "dataset/rand_idx.npz"
    alphas = [1., 0.9, 0.8, 0.7, 0.6]
    window_sizes = [1, 3, 5, 10, 20]


    preprocessor = DataProcessor(overwrite=True, shuffle=True)
    preprocessor.run_docs(f_corpus=f_corpus, f_meta_data=f_meta_data, f_dataset_out=f_dataset_out,
                          f_vocab=f_vocab, f_sidx=f_sidx)
    preprocessor.save_labels(f_labels_out=f_labels)
    preprocessor.save_sidx(f_sidx_out=f_sidx+"2")

    preprocessor.run_lda(dir_lda=dir_lda, f_lda_out=f_lda_out,
                         alphas=alphas, window_sizes=window_sizes,
                         f_meta_data=f_meta_data)
