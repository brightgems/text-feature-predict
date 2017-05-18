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
    :return: corpus_coverage: dict of {company_name -> [dates, line_number]}
    """
    with open(f_corpus, "r") as f:
        lines = f.readlines()

    corpus_coverage = dict()
    for i, line in enumerate(lines):
        line = line.strip().split("\t")
        if line[0] in corpus_coverage:
            corpus_coverage[line[0]].append((line[1], i))
        else:
            corpus_coverage[line[0]] = [(line[1], i)]

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


def get_labels(x, sig_pctg=1):
    if abs(x) >= sig_pctg:
        if x > 0:
            return 1
        else:
            return 0
    else:
        print 'ERROR: Cant generate label for abs({}) < {}'.format(x, sig_pctg)


def sig_statis(stock_dir, corpus_coverage, map_sym_comp, labels_out, sig_pctg=1, num_previous_changes=0,
               use_stock_prices=False):
    corpus_sig = dict()

    l_out = open(labels_out, 'w')

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
            doc_dates = dict()
            doc_tuples = corpus_coverage[company]
            for doc_tuple in doc_tuples:
                doc_dates[doc_tuple[0]] = doc_tuple[1]
        else:
            #print "unfound company in corpus:", company
            continue

        # get significance label
        stock_file = stock_dir + stock_file
        df = pd.read_csv(stock_file)
        df['Open_1'] = df['Open'].shift(+1)
        df['Change_per'] = (df['Open_1'] - df['Open']) / df['Open']
        df['Change_per'] = df.apply(lambda row: (row['Change_per']) * 100, axis=1)
        df['sig?'] = df.apply(lambda row: significant_change(row['Change_per'], sig_pctg), axis=1)

        for i in range(num_previous_changes):
            i += 1
            df['Change_per_{}'.format(i)] = df['Change_per'].shift(-i)
            df['Change_per_{}_Date'.format(i)] = df['Date'].shift(-i)

        for i in range(num_previous_changes):
            df['Open_hist_{}'.format(i)] = df['Open'].shift(-i)

        # print df[20:30]

        # df = df.ix[:, ['Date', 'Change_per', 'sig?']]
        cnt_sig = 0
        for idx, row in df.iterrows():
            if row["Date"] in doc_dates.keys() and row["sig?"] == 1:
                cnt_sig += 1
                label = get_labels(row["Change_per"], sig_pctg=1)
                if use_stock_prices:
                    l_out.write("{},{},{},{}".format(
                        company, row["Date"], doc_dates[row["Date"]], row["Open_1"]))
                    for i in range(num_previous_changes):
                        l_out.write(",{}".format(row['Open_hist_{}'.format(i)]))
                else:
                    l_out.write("{},{},{},{}".format(
                        company, row["Date"], doc_dates[row["Date"]], row["Change_per"]))
                    for i in range(num_previous_changes):
                        i += 1
                        l_out.write(",{}".format(row['Change_per_{}'.format(i)]))
                    for i in range(num_previous_changes):
                        i += 1
                        date = row['Change_per_{}_Date'.format(i)]
                        date = doc_dates[date] if date in doc_dates else -1
                        l_out.write(",{}".format(date))

                l_out.write(",{}\n".format(label))
        corpus_sig[company] = cnt_sig
        print company, cnt_sig

    total_sig = sum([corpus_sig[kk] for kk, vv in corpus_sig.iteritems()])
    print "==================================="
    print "threshold:", sig_pctg
    print "total significant records:", total_sig
    print "==================================="

    l_out.close()

if __name__ == "__main__":
    data_path = "/Users/ds/git/financial-topic-modeling/data/bpcorpus/"
    stock_dir = data_path + "stock-price-data/"
    f_corpus = data_path + "standard-query-corpus_pp.tsv"
    f_stock_mapping = data_path + "nasdaq-100.csv"
    f_labels_out = data_path + "corpus_labels.csv"
    num_previous_changes = 20

    sig_pctg = 1.

    map_sym_comp = load_mapping(f_stock_mapping=f_stock_mapping)
    corpus_coverage = load_corpus(f_corpus=f_corpus)

    sig_statis(stock_dir=stock_dir, corpus_coverage=corpus_coverage, map_sym_comp=map_sym_comp, labels_out=f_labels_out,
               sig_pctg=sig_pctg, num_previous_changes=num_previous_changes)
