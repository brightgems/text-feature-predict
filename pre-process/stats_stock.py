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
    """
    Get the label for a significant price change. 1, if significant; 0, otherwise
    :param x: price
    :param sig_pctg: significance threshold
    :return:
    """
    if abs(x) >= sig_pctg:
        return 1
    else:
        return 0


def insignificant_change(x, insig_pctg=0.05):
    """
    Get the label for an insignificant price change. 1, if insignificant; 0, otherwise
    :param x: price
    :param insig_pctg: insignificance threshold
    """
    if abs(x) <= insig_pctg:
        return 1
    else:
        return 0


def get_labels_trend(x, sig_pctg=1):
    """
    Get the label for the price trend. 1, if trend positive; 0, if trend negative
    :param x: price
    :param sig_pctg: percentage threshold
    :return: label
    """
    if abs(x) >= sig_pctg:
        if x > 0:
            return 1
        else:
            return 0
    else:
        print 'ERROR: Cant generate label for abs({}) < {}'.format(x, sig_pctg)


def combine_stock_label(stock_dir, corpus_coverage, map_sym_comp, labels_out, sig_pctg=1, insig_pctg=None,
                        num_previous_days=0, use_stock_prices=False):
    """
    Combine the stock data with the label information. Datapoints that do not have a corresponding document in the
    corpus are filtered out.

    :param stock_dir: Path to folder with stock price data
    :param corpus_coverage: Information about which dates are covered in corpus
    :param map_sym_comp: Mapping of stock symbols (AAPL) to company names (Apple Inc.)
    :param labels_out: Output file
    :param sig_pctg: Threshold for significant price change
    :param insig_pctg: Threshold for insignificant price change, None if insignificant irrelevant
    :param num_previous_days: Number of previous days of stock information
    :param use_stock_prices: Use stock prices instead of stock changes
    """

    corpus_sig = dict()

    l_out = open(labels_out, 'w')

    for stock_file in listdir(stock_dir):
        # parse to get company name
        symbol = re.sub(r'.*_', '', stock_file)
        symbol = symbol.replace('.csv', '')
        if symbol in map_sym_comp:
            company = map_sym_comp[symbol]
        else:
            continue

        # get dates with valid documents
        if company in corpus_coverage:
            doc_dates = dict()
            doc_tuples = corpus_coverage[company]
            for doc_tuple in doc_tuples:
                doc_dates[doc_tuple[0]] = doc_tuple[1]
        else:
            continue

        # get significance label
        stock_file = stock_dir + stock_file
        df = pd.read_csv(stock_file)
        df['Open_1'] = df['Open'].shift(+1)
        df['Change_per'] = (df['Open_1'] - df['Open']) / df['Open']
        df['Change_per'] = df.apply(lambda row: (row['Change_per']) * 100, axis=1)
        df['sig?'] = df.apply(lambda row: significant_change(row['Change_per'], sig_pctg), axis=1)
        if insig_pctg is not None:
            df['insig?'] = df.apply(lambda row: insignificant_change(row['Change_per'], insig_pctg), axis=1)

        for i in range(num_previous_days):
            i += 1
            df['Change_per_{}'.format(i)] = df['Change_per'].shift(-i)
            df['Change_per_{}_Date'.format(i)] = df['Date'].shift(-i)

        for i in range(num_previous_days):
            df['Open_hist_{}'.format(i)] = df['Open'].shift(-i)

        # print df[20:30]

        # df = df.ix[:, ['Date', 'Change_per', 'sig?']]
        cnt_sig = 0
        cnt_insig = 0
        for idx, row in df.iterrows():
            if row["Date"] in doc_dates.keys():
                if insig_pctg is None:
                    if row["sig?"] == 1:
                        label = get_labels_trend(row["Change_per"], sig_pctg=1)
                        cnt_sig += 1
                    else:
                        continue
                else:
                    if row["sig?"] == 1 or row["insig?"] == 1:
                        if row["sig?"] == 1:
                            label = 1
                            cnt_sig += 1
                        else:
                            label = 0
                            cnt_insig += 1
                    else:
                        continue



                if use_stock_prices:
                    l_out.write("{},{},{},{}".format(
                        company, row["Date"], doc_dates[row["Date"]], row["Open_1"]))
                    for i in range(num_previous_days):
                        l_out.write(",{}".format(row['Open_hist_{}'.format(i)]))
                else:
                    l_out.write("{},{},{},{}".format(
                        company, row["Date"], doc_dates[row["Date"]], row["Change_per"]))
                    for i in range(num_previous_days):
                        i += 1
                        l_out.write(",{}".format(row['Change_per_{}'.format(i)]))
                    for i in range(num_previous_days):
                        i += 1
                        date = row['Change_per_{}_Date'.format(i)]
                        date = doc_dates[date] if date in doc_dates else -1
                        l_out.write(",{}".format(date))

                l_out.write(",{}\n".format(label))
        corpus_sig[company] = cnt_sig, cnt_insig
        print company, cnt_sig, cnt_insig

    total_sig = sum([corpus_sig[kk][0] for kk, vv in corpus_sig.iteritems()])

    if insig_pctg is not None:
        total_insig = sum([corpus_sig[kk][1] for kk, vv in corpus_sig.iteritems()])

    print "==================================="
    print "significance threshold:", sig_pctg
    print "total significant records:", total_sig
    if insig_pctg is not None:
        print "insignificance threshold:", insig_pctg
        print "total insignificant records:", total_insig
    print "==================================="

    l_out.close()

if __name__ == "__main__":
    data_path = "/Users/ds/git/financial-topic-modeling/data/bpcorpus/"
    stock_dir = data_path + "stock-price-data/"
    f_corpus = data_path + "standard-query-corpus_pp.tsv"
    f_stock_mapping = data_path + "nasdaq-100.csv"
    f_labels_out = data_path + "corpus_labels_insignificant.csv"
    num_previous_changes = 20

    sig_pctg = 1.
    insig_pctg = 0.5

    map_sym_comp = load_mapping(f_stock_mapping=f_stock_mapping)
    corpus_coverage = load_corpus(f_corpus=f_corpus)

    combine_stock_label(stock_dir=stock_dir, corpus_coverage=corpus_coverage, map_sym_comp=map_sym_comp,
                        labels_out=f_labels_out, sig_pctg=sig_pctg, insig_pctg=insig_pctg,
                        num_previous_days=num_previous_changes, use_stock_prices=False)
