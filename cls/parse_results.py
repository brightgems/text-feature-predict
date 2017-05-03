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


if __name__ == '__main__':
    results = 'results/lda.txt'
    print 'lda today:'
    lda_today(results)

    print '\nks for hist.add, hist.cont:'
    hist_cont_add(results)

    print '\nhist.add:'
    lda_hist_add(results)

    print '\nhist.cont:'
    lda_hist_cont(results)
