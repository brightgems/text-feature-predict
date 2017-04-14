"""
prepare crawled data for feature extraction
take both json and text file as input
extract date and textual article
"""

import ast # abstract syntax trees
import os
import numpy
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

class DataProcessor:
    def __init__(self, f_corpus, f_meta_data, f_dataset_out, f_vocab=None,
                 vocab_size=None, valid_portion=None, overwrite=False):
        self.train = DataPoints()
        self.valid = DataPoints()
        self.test = DataPoints()
        self.vocab = dict() # vocab for the loaded dataset
        self.vocab_size = vocab_size # take top vocab_size vocab for loaded dataset if not None

        # preprocess
        def _run():
            self.load_data(f_corpus, f_meta_data, shuffle=True)
            self.gen_vocab(f_vocab=f_vocab)
            self.save_data(f_dataset_out=f_dataset_out)

        try:
            os.stat(f_dataset_out)
            print f_dataset_out + " already exist!"
            if overwrite:
                print "overwriting ..."
                _run()
        except:
            _run()

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
        wordlist = _get_vocab(self.train.x, wordlist)
        wordlist = _get_vocab(self.test.x, wordlist)

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

    def load_data(self, f_corpus, f_meta_data, shuffle=True):
        """
        load data from corpus and corpus mapping file
        :param f_corpus: corpus {company, date, docs}, tap separated
        :param f_meta_data: meta data, comma separated
                            {company, date, line index in corpus (starting 0), label, train(0)/test(1)}
        """
        print "loading from {}".format(f_corpus),
        self.train.clear()
        self.test.clear()

        with open(f_meta_data, "r") as f:
            meta_data = f.readlines()
        with open(f_corpus, "r") as f:
            corpus = f.readlines()

        for lidx, meta_line in enumerate(meta_data):
            meta_line = meta_line.strip().split(",")
            if len(meta_line) != 5:
                print "warning: line {} has less than three columns".format(lidx+1)
                continue
            doc = word_tokenize(corpus[int(meta_line[2])].strip().split("\t")[-1]) # get doc from corpus
            label = int(meta_line[3])
            if label == 0:
                label = -1
            if int(meta_line[-1]) == 0:
                self.train.x.append(doc) # text
                self.train.y.append(label) # label
            elif int(meta_line[-1]) == 1:
                self.test.x.append(doc)
                self.test.y.append(label)
            else:
                print "warning: fail to recognize train/test label {0} at line {1}".format(meta_line[1], lidx)

        if shuffle:
            self.shuffle()
        print "done!"

    def shuffle(self):
        print "using shuffling...",
        sidx = numpy.random.permutation(len(self.train.x))
        self.train.x = [self.train.x[idx] for idx in sidx]
        self.train.y = [self.train.y[idx] for idx in sidx]
        sidx = numpy.random.permutation(len(self.test.x))
        self.test.x = [self.test.x[idx] for idx in sidx]
        self.test.y = [self.test.y[idx] for idx in sidx]

    def set_valid(self, valid_portion=0.15):
        """
        set valid_portion of training data into validation set
        """
        n_sample = len(self.train.x)
        sidx = numpy.random.permutation(n_sample)
        n_train = int(numpy.round(n_sample * (1 - valid_portion)))
        self.valid.x = [self.train.x[idx] for idx in sidx[n_train:]]
        self.valid.y = [self.train.y[idx] for idx in sidx[n_train:]]
        self.train.x = [self.train.x[idx] for idx in sidx[:n_train]]
        self.train.y = [self.train.y[idx] for idx in sidx[:n_train]]

    def save_data(self, f_dataset_out):
        print "saving to file ...",
        train = [[" ".join(line) for line in self.train.x], self.train.y]
        test = [[" ".join(line) for line in self.test.x], self.test.y]
        valid = [[" ".join(line) for line in self.valid.x], self.valid.y]
        with open(f_dataset_out, "wb") as f:
            pkl.dump(train, f)
            pkl.dump(test, f)
            pkl.dump(valid, f)
        print "done!"
        print "train:", len(self.train.x), "valid:", len(self.valid.x), "test:", len(self.test.x)


if __name__ == "__main__":
    dir_data = "/home/yiren/Documents/time-series-predict/data/bp/"
    f_corpus = dir_data + "standard-query-corpus_pp.tsv"
    f_meta_data = dir_data + "corpus_lablels_split.csv"
    f_dataset_out = dir_data + "dataset/corpus_bp_cls.npz"
    f_vocab = dir_data + "dataset/vocab.npz"

    preprocessor = DataProcessor(f_corpus=f_corpus,
                                 f_meta_data=f_meta_data,
                                 f_dataset_out=f_dataset_out,
                                 f_vocab=f_vocab,
                                 overwrite=True)
