"""
statistical information regarding the dataset
"""
from os import listdir
import re
import pandas as pd

def load_mapping(f_stock_mapping):
    """
    get stock_symbol and company_name mapping
    :param f_stock_mapping: mapping file
    :return: map_sym_comp: dict [symbol -> company name]
    """
    df_stock = pd.read_csv(f_stock_mapping, sep=",", header=None,
                           names=["symbol", "company_name", "val", "total", "year", "service", "category", "web"])
    df_stock = df_stock.ix[:, ['symbol', 'company_name']]
    map_tmp = df_stock.set_index('symbol').T.to_dict("list")

    map_sym_comp = dict()
    for kk, vv in map_tmp.iteritems():
        if vv[0][-1] == " ":
            vv[0] = vv[0][:-1]
        map_sym_comp[kk] = vv[0]
    print "number of companies:", len(map_sym_comp)
    return map_sym_comp

def load_corpus(f_corpus):
    """
    load corpus
    :param f_corpus: [company_name, date, document]
    :return: corpus_coverage: dict of {company_name -> [dates]}
    """
    with open(f_corpus, "r") as f:
        lines = f.readlines()

    corpus_coverage = dict()
    for line in lines:
        line = line.strip().split("\t")
        if line[0] in corpus_coverage:
            corpus_coverage[line[0]].append(line[1])
        else:
            corpus_coverage[line[0]] = [line[1]]

    company_num = len(corpus_coverage)
    records_num = sum([len(corpus_coverage[kk]) for kk, vv in corpus_coverage.iteritems()])

    print "number of companies:", company_num
    print "number of records:", records_num
    return corpus_coverage

def significant_change(x, sig_pctg=1):
    if abs(x) >= sig_pctg:
        return 1
    else:
        return 0

def sig_statis(stock_dir, corpus_coverage, map_sym_comp, sig_pctg=1):
    corpus_sig = dict()

    for stock_file in listdir(stock_dir):
        # parse to get company name
        symbol = re.sub(r'.*_', '', stock_file)
        symbol = symbol.replace('.csv', '')
        if symbol in map_sym_comp:
            company = map_sym_comp[symbol]
        else:
            #print "unfound stock symbol in mapping:", symbol
            continue

        # get dates with valid documents
        if company in corpus_coverage:
            doc_dates = corpus_coverage[company]
        else:
            #print "unfound company in corpus:", company
            continue

        # get significance label
        stock_file = stock_dir + stock_file
        df = pd.read_csv(stock_file)
        df['Close_1'] = df['Close'].shift(+1)
        df['Change_per_1'] = df['Close_1'] / df['Close_1'].shift(-1)
        df['Change_per_1'] = df.apply(lambda row: (1 - row['Change_per_1']) * -100, axis=1)
        df['sig?'] = df.apply(lambda row: significant_change(row['Change_per_1'], sig_pctg), axis=1)
        # df['sig_mc?'] = df.apply(lambda row: significant_change_multiclass(row['Change_per_1']), axis=1)

        df = df.ix[:, ['Date', 'sig?']]
        cnt_sig = 0
        for idx, row in df.iterrows():
            if row["Date"] in doc_dates and row["sig?"] == 1:
                cnt_sig += 1
        corpus_sig[company] = cnt_sig
        print company, cnt_sig

    total_sig = sum([corpus_sig[kk] for kk, vv in corpus_sig.iteritems()])
    print "==================================="
    print "threshold:", sig_pctg
    print "total significant records:", total_sig
    print "==================================="

if __name__ == "__main__":
    data_path = "/Users/Irene/Documents/financial_topic_model/data/bpcorpus/"
    stock_dir = data_path + "stock-price-data/"
    f_corpus = data_path + "standard-query-corpus.tsv"
    f_stock_mapping = data_path + "nasdaq-100.csv"

    sig_pctg = 1.

    map_sym_comp = load_mapping(f_stock_mapping=f_stock_mapping)
    corpus_coverage = load_corpus(f_corpus=f_corpus)

    sig_statis(stock_dir=stock_dir, corpus_coverage=corpus_coverage, map_sym_comp=map_sym_comp, sig_pctg=sig_pctg)