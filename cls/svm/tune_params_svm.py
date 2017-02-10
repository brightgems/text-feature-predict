# used to tune parameters using liblinear logistic regression
# usage: python TuneParameters train_file valid_file use_scale_or_not
# Highly recommend scale it

import os
import copy
import time
import random
import sys
import shutil
import subprocess

from subprocess import STDOUT

kernels = [0, 1, 2] 
kernel_name = ['linear', 'polynomial', 'rbf', 'sigmoid']
Cparas = [2**i for i in range(-5, 10)]

# Execute a subprocess with standard output
def execute(command, output = False):
    popen = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    lines_iterator = iter(popen.stdout.readline, b"")
    returned_result = ''
    for line in lines_iterator:
        if(output):
            print(line) # yield line
        returned_result += line
    return returned_result

if __name__=="__main__":
    liblinear_train = './svm-train'
    liblinear_test = './svm-predict'
    svm_scale = './svm-scale'

    train_file = sys.argv[1]
    valid_file = sys.argv[2]
    scale = len(sys.argv) >= 4 and sys.argv[3] == '1'
    final_result_output = '{0}.tune.log'.format(train_file)
    fout = open(final_result_output, 'w')

    if scale:
        scale_paras = ' -s train.scale.para {0} > train.scale'.format(train_file)
        execute(svm_scale + scale_paras)
        scale_paras = ' -r train.scale.para {0} > valid.scale'.format(valid_file)
        execute((svm_scale + scale_paras))
        train_file = 'train.scale'
        valid_file = 'valid.scale'

    for kernel in kernels:

        model_file = kernel_name[kernel]+'.model.tune'

        for i in xrange(len(Cparas)):
            sys.stdout.write('{0} Kernel, C paras:{1} '.format(kernel_name[kernel], Cparas[i]))
            fout.write('{0} Kernel, C paras:{1} '.format(kernel_name[kernel], Cparas[i]))
            train_paras = ' -s 0 -t {0} -c {1} -q {2} {3}'.format(kernel, Cparas[i], train_file, model_file)
            execute(liblinear_train + train_paras)
            valid_paras = ' {0} {1} valid_predict.output.txt'.format(valid_file, model_file)
            result = execute(liblinear_test + valid_paras, True)
            fout.write(result)

    fout.close()

