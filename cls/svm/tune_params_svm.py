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

kernels = [0, 2]
kernel_name = ['linear', 'polynomial', 'rbf', 'sigmoid']
Cparas = [2**i for i in range(-5, 10)]
Gparas = [2**i for i in range(-10, 3)]

# Execute a subprocess with standard output
def execute(command, output=False):
    popen = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    lines_iterator = iter(popen.stdout.readline, b"")
    returned_result = ''
    for line in lines_iterator:
        if output:
            print line # yield line
        returned_result += line
    return returned_result

def parse_result(result):
    return float(result.split("= ")[-1].split("%")[0])

if __name__=="__main__":
    liblinear_train = './svm-train'
    liblinear_test = './svm-predict'
    svm_scale = './svm-scale'

    if len(sys.argv) < 3:
        print "$train_file $valid_file $scale(set to 1 if use scale) $kernel(use default if not set)"
        print "kernel: 0--linear, 1--polynomial, 2--rbf, 3--sigmoid, default: [0,2]"

    train_file = sys.argv[1]
    valid_file = sys.argv[2]
    scale = len(sys.argv) >= 4 and sys.argv[3] == '1' # whether to scale
    if len(sys.argv) >= 5:
        print "using kernel:", kernel_name[int(sys.argv[4])]
        svm_kernels = [int(sys.argv[4])] # set kernel
    else:
        print "using default kernel candidate:", kernels
        svm_kernels = kernels

    try:
        os.stat("tune_logs")
    except:
        os.mkdir("tune_logs")
    final_result_output = 'tune_logs/{0}.tune.log'.format(train_file.split("/")[-1])
    fout = open(final_result_output, 'w')

    if scale:
        print "using scale"
        scale_paras = ' -s train.scale.para {0} > train.scale'.format(train_file)
        execute(svm_scale + scale_paras)
        scale_paras = ' -r train.scale.para {0} > valid.scale'.format(valid_file)
        execute((svm_scale + scale_paras))
        train_file = 'train.scale'
        valid_file = 'valid.scale'

    best_model = {"accu": 0.0, "kernel": -1, "c": -1, "g": -1}

    for kernel in svm_kernels:
        model_file = kernel_name[kernel]+'.model.tune'
        if kernel == 2: # tune both c and g
            for c in Cparas:
                for g in Gparas:
                    # sys.stdout.write('Kernel:{0}, C paras:{1}, Gamma:{2}'.format(kernel_name[kernel], c, g))
                    fout.write('Kernel:{0}, C paras:{1}, Gamma:{2}'.format(kernel_name[kernel], c, g))
                    train_paras = ' -s 0 -t {0} -c {1} -g {2} -q {3} {4}'.format(kernel, c, g, train_file, model_file)
                    execute(liblinear_train + train_paras)
                    valid_paras = ' {0} {1} valid_predict.output.txt'.format(valid_file, model_file)
                    result = execute(liblinear_test + valid_paras, False)
                    fout.write(result)
                    accu = parse_result(result)
                    if accu > best_model["accu"]:
                        best_model["accu"] = accu
                        best_model["kernel"] = kernel
                        best_model["c"] = c
                        best_model["g"] = g
        else: # tune c
            for c in Cparas:
                # sys.stdout.write('Kernel:{0}, C paras:{1} '.format(kernel_name[kernel], c))
                fout.write('Kernel:{0}, C paras:{1} '.format(kernel_name[kernel], c))
                train_paras = ' -s 0 -t {0} -c {1} -q {2} {3}'.format(kernel, c, train_file, model_file)
                execute(liblinear_train + train_paras)
                valid_paras = ' {0} {1} valid_predict.output.txt'.format(valid_file, model_file)
                result = execute(liblinear_test + valid_paras, False)
                fout.write(result)
                accu = parse_result(result)
                if accu > best_model["accu"]:
                    best_model["accu"] = accu
                    best_model["kernel"] = kernel
                    best_model["c"] = c
    print "Best model: Kernel:{0}, C:{1}, Gamma:{2}, Accuracy:{3}\n".format(best_model["kernel"],
                                                                            best_model["c"], best_model["g"],
                                                                            best_model["accu"])
    fout.close()

