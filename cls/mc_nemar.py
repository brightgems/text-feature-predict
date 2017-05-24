true_labels = []
time_series = []
topic_model = []
ngram_model = []

def parse(line):
    return [int(num) for num in line.replace('[', '').replace(']', '').strip().split(', ')]



for line in open('results/statistical-significance.txt'):
    if line.startswith('true labels:'):
        line = line.replace('true labels: ', '')
        true_labels = parse(line)
    if line.startswith('time-series:'):
        line = line.replace('time-series: ', '')
        time_series = parse(line)
    if line.startswith('topic-model:'):
        line = line.replace('topic-model: ', '')
        topic_model = parse(line)
    if line.startswith('ngram-model:'):
        line = line.replace('ngram-model: ', '')
        ngram_model = parse(line)

b = 0
c = 0

for i in range(len(true_labels)):
    if time_series[i] == true_labels[i] and topic_model[i] != true_labels[i]:
        b += 1
    elif time_series[i] != true_labels[i] and topic_model[i] == true_labels[i]:
        c += 1

print 'X^2 Time-Series -> Topic = ', pow((b-c), 2) / float(b+c)

b = 0
c = 0

for i in range(len(true_labels)):
    if time_series[i] == true_labels[i] and ngram_model[i] != true_labels[i]:
        b += 1
    elif time_series[i] != true_labels[i] and ngram_model[i] == true_labels[i]:
        c += 1

print 'X^2 Time-Series -> N-grams = ', pow((b-c), 2) / float(b+c)
