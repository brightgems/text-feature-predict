import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

class DataReader:
    def __init__(self, f_corpus_labels_stock_price):
        self.x_train = None
        self.x_test = None
        self.y_train_price = None
        self.y_test_price = None
        self.y_train_labels = None
        self.y_test_labels = None

        self.load_data(f_corpus_labels_stock_price)

    def load_data(self, f_corpus_labels_stock_price):
        print 'loading data from {}...'.format(f_corpus_labels_stock_price),

        x_train = []
        x_test = []
        y_train_price = []
        y_test_price = []
        y_train_labels = []
        y_test_labels = []

        for line in open(f_corpus_labels_stock_price):
            line = line.strip().split(',')
            line[-1] = int(line[-1])
            if line[-1] == 0:
                x_train.append(self.parse_float(line[-22:-2]))
                y_train_price.append(float(line[3]))
                y_train_labels.append(int(line[-2]))
            if line[-1] == 1:
                x_test.append(self.parse_float(line[-22:-2]))
                y_test_price.append(float(line[3]))
                y_test_labels.append(int(line[-2]))

        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train_price = np.array(y_train_price)
        self.y_test_price = np.array(y_test_price)
        self.y_train_labels = np.array(y_train_labels)
        self.y_test_labels = np.array(y_test_labels)


        print 'done!'
        print '{} training, {} testing instances.'.format(len(self.x_train), len(self.x_test))

    def parse_float(self, line):
        return [float(token) if token != 'nan' else 0 for token in line]

class VAR:
    def __init__(self, data_reader):
        self.data_reader = data_reader
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.predicted = None
        self.predicted_labels = None
        self.sig_prct = 1

        # set y train/test
        self.y_train = self.data_reader.y_train_price
        self.y_test = self.data_reader.y_test_price
        self.predicted_labels = np.zeros(shape=(len(self.y_test), 1))

    def run_var(self, num_days):

        for num_day in num_days:
            print 'performing regression for {} days... '.format(num_day),
            self.x_train = self.data_reader.x_train[:, :num_day+1]
            self.x_test = self.data_reader.x_test[:, :num_day+1]
            self.run_linear_regression()
            self.evaluate()
            # print 'done!'

    def evaluate(self):
        for i in range(len(self.x_test)):
            predicted_price_change = (self.predicted[i] - self.x_test[i][0]) * 100 / self.x_test[i][0]
            actual_price_change = (self.y_test[i] - self.x_test[i][0]) * 100 / self.x_test[i][0]
            self.predicted_labels[i] = -1
            if predicted_price_change > 0:
                self.predicted_labels[i] = 1
            elif predicted_price_change <= 0:
                self.predicted_labels[i] = 0
            if predicted_price_change == 0:
                print '[warning] predicted price change is 0'
            # print 'price pred: {}, price act: {}, label pred: {}, label act: {},' \
            #       ' price change pred: {}, price change act: {}'.format(
            #     self.predicted[i], self.y_test[i], predicted_label, data_reader.y_test_labels[i],
            #     predicted_price_change, actual_price_change)
        print accuracy_score(y_true=data_reader.y_test_labels, y_pred=self.predicted_labels)

    def run_linear_regression(self):
        cls_model = LinearRegression()
        cls_model.fit(self.x_train, self.y_train)
        self.predicted = cls_model.predict(self.x_test)


if __name__ == '__main__':
    dir_data = '/Users/ds/git/financial-topic-modeling/data/bpcorpus/'
    f_corpus_labels_stock_price = dir_data + "corpus_labels_stock_price_split.csv"

    data_reader = DataReader(f_corpus_labels_stock_price=f_corpus_labels_stock_price)

    num_days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    myModel = VAR(data_reader=data_reader)
    myModel.run_var(num_days)

