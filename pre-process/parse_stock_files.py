import pandas as pd
import csv
import re
from os import listdir

stock_folder = 'data/stock-prices/'
company_stock_file = 'data/company-stock-mapping.csv'
output_file = 'data/stock-data.csv'
sig_perc = 2
sig_perc_multic = 1


def significant_change(x, sig_perc=2):
    if abs(x) >= sig_perc:
        return 1
    else:
        return 0


def significant_change_multiclass(x):
    if abs(x) >= sig_perc_multic:
        if x > 0:
            return 1
        else:
            return -1
    else:
        return 0


def company_stock_dictionary(path):
    with open(path) as f:
        f.readline()  # ignore first line (header)
        dictionary = dict(csv.reader(f, delimiter=','))

    return dictionary


if __name__ == '__main__':

    cs_dict = company_stock_dictionary(company_stock_file)

    stock_data = pd.DataFrame(columns=['Date', 'Company', 'Change_per_1', 'sig?'])

    # print stock_data

    for stock_file in listdir(stock_folder):
        company = re.sub(r'.*_', '', stock_file)
        company = company.replace('.csv', '')

        stock_file = stock_folder + stock_file
        f = open(stock_file, 'r')
        df = pd.DataFrame.from_csv(f, index_col=False)

        df['Close_1'] = df['Close'].shift(+1)

        df['Change_per_1'] = df['Close_1'] / df['Close_1'].shift(-1)
        df['Change_per_1'] = df.apply(lambda row: (1 - row['Change_per_1']) * -100, axis=1)

        df['sig?'] = df.apply(lambda row: significant_change(row['Change_per_1']), axis=1)

        df['sig_mc?'] = df.apply(lambda row: significant_change_multiclass(row['Change_per_1']), axis=1)

        df['Company'] = df.apply(lambda row: cs_dict[company], axis=1)

        # print df[:10]

        stock_data = stock_data.append(df[['Date', 'Company', 'Close_1', 'Change_per_1', 'sig?', 'sig_mc?']], ignore_index=True)

    print stock_data[:10]

    stock_data.to_csv(output_file, index_label=False, index=False)