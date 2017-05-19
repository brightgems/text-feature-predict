import re


def lda_hist_add(f_results):
    content = dict()
    for line in open(f_results):
        if 'alpha' in line and 'cond' not in line:
            k, alpha, l, acc_test = get_k_alpha_l_acc(line)
            if alpha not in content:
                content[alpha] = dict()
            if l not in content[alpha]:
                content[alpha][l] = dict()
            content[alpha][l][k] = acc_test

    print_lda_hist(content)


def lda_hist_cont(f_results):
    content = dict()
    for line in open(f_results):
        if 'alpha' in line and 'cond' in line:
            k, alpha, l, acc_test = get_k_alpha_l_acc(line)
            if alpha not in content:
                content[alpha] = dict()
            if l not in content[alpha]:
                content[alpha][l] = dict()
            content[alpha][l][k] = acc_test

    print_lda_hist(content)


def print_lda_hist(content):
    alphas = content.keys()
    alphas.sort()
    ls = content[alphas[0]].keys()
    ls.sort()

    print '\\hline'
    print '$\\alpha$',
    for l in ls:
        print '&', '$L={}$'.format(l),
    print '\\\\'
    print '\\hline'

    for alpha in alphas:
        print alpha,
        for l in ls:
            acc = max(content[alpha][l].values())
            print '&', '%0.2f' % (acc * 100),
        print '\\\\'


def lda_today(f_results):
    content = dict()

    for line in open(f_results):
        if 'today' in line:
            k = int(re.findall(r'k=..', line)[0].replace('k=', ''))
            acc_test = float(re.findall(r'test: .*$', line)[0].replace('test: ', ''))
            content[k] = acc_test

    ks = content.keys()
    ks.sort()
    for k in ks:
        print k, '&', '%0.2f' % (content[k] * 100), '&', '&', '&', '&', '&', '&', '\\\\'


def lda_change(f_results):
    content_dist = dict()
    content_change = dict()

    for line in open(f_results):
        k = int(re.findall(r'k=..', line)[0].replace('k=', ''))
        acc_test = float(re.findall(r'test: .*$', line)[0].replace('test: ', ''))
        if 'topic distribution' in line:
            content_dist[k] = acc_test
        elif 'change only' in line:
            content_change[k] = acc_test

    ks = content_change.keys()
    ks.sort()
    print 'change only & change + topic dist'
    for k in ks:
        print k, '%0.2f' % (content_change[k] * 100), '&', '%0.2f' % (content_dist[k] * 100)

def lda_change_history_add(f_results):
    content = dict()

    for line in open(f_results):
        if 'cond' not in line:
            k = int(re.findall(r'k=..', line)[0].replace('k=', ''))
            acc_test = float(re.findall(r'test: .*$', line)[0].replace('test: ', ''))
            if k not in content:
                content[k] = []
            content[k].append(acc_test)

    ks = content.keys()
    ks.sort()
    for k in ks:
        print k, '&', '%0.2f' % (max(content[k]) * 100)


def lda_change_history_cont(f_results):
    content = dict()

    for line in open(f_results):
        if 'cond' in line:
            k = int(re.findall(r'k=..', line)[0].replace('k=', ''))
            acc_test = float(re.findall(r'test: .*$', line)[0].replace('test: ', ''))
            if k not in content:
                content[k] = []
            content[k].append(acc_test)

    ks = content.keys()
    ks.sort()
    for k in ks:
        print k, '&', '%0.2f' % (max(content[k]) * 100)


def hist_cont_add(f_results):
    content_add = dict()
    content_cont = dict()
    for line in open(f_results):
        if 'alpha' in line:
            k, alpha, l, acc_test = get_k_alpha_l_acc(line)
            if 'cond' not in line:
                if k not in content_add:
                    content_add[k] = list()
                content_add[k].append(acc_test)
            if 'cond' in line:
                if k not in content_cont:
                    content_cont[k] = list()
                content_cont[k].append(acc_test)

    ks = content_add.keys()
    ks.sort()
    for k in ks:
        print k, '%0.2f' % (max(content_add[k]) * 100), '&', \
            '%0.2f' % (max(content_cont[k]) * 100)


def get_k_alpha_l_acc(line):
    k = int(re.findall(r'k=..', line)[0].replace('k=', ''))
    alpha = float(re.findall(r'alpha=...', line)[0].replace('alpha=', ''))
    l = int(re.findall(r'L=..?', line)[0].replace('L=', ''))
    acc_test = float(re.findall(r'test: .*$', line)[0].replace('test: ', ''))

    return k, alpha, l, acc_test


def stock_history(f_results):
    accs = []
    for i, line in enumerate(open(f_results).readlines()):
        accs.append(float(re.findall(r'test: .*$', line)[0].replace('test: ', '')))

    for i, acc in enumerate(accs):
        print i+1, '&', '%0.2f' % (acc * 100), '\\\\'

if __name__ == '__main__':
    # results_lda = 'results/lda_insignificant.txt'
    # print 'lda today:'
    # lda_today(results_lda)
    #
    # print '\nks for hist.add, hist.cont:'
    # hist_cont_add(results_lda)
    #
    # print '\nhist.add:'
    # lda_hist_add(results_lda)
    #
    # print '\nhist.cont:'
    # lda_hist_cont(results_lda)

    # results_stock_history = 'results/stock_history.txt'
    #
    # stock_history(results_stock_history)

    # results_lda_change_history = 'results/lda_change_history_insignificant.txt'
    # print 'history add + change'
    # lda_change_history_add(results_lda_change_history)
    #
    # print '\nhistory cont + change'
    # lda_change_history_cont(results_lda_change_history)

    results_lda_change = 'results/lda_change_insignificant.txt'
    lda_change(results_lda_change)