# used to tune parameters for epsilon-support vector regression
# usage: python TuneParameters train_file valid_file use_scale_or_not
# Highly recommend scale it
# tuning c, epsilon, kernel, gamma, shrinking

import os
import copy
import time
import random
import sys
import shutil
import subprocess

from subprocess import STDOUT

kernels = [2]
kernel_name = ['linear', 'polynomial', 'rbf', 'sigmoid']
Cparas = [2**i for i in range(-5, 10)]
Gparas = [2**i for i in range(-10, 3)]
Epsilon = [0, 0.01, 0.1, 0.5, 1, 2, 4, 8]


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
    lines = result.split("\n")
    mse = float(lines[0].split("= ")[-1].split(" (")[0])
    scc = float(lines[1].split("= ")[-1].split(" (")[0])
    return mse, scc

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
        print "\tusing kernel:", kernel_name[int(sys.argv[4])]
        svm_kernels = [int(sys.argv[4])] # set kernel
    else:
        print "\tusing default kernel candidate:", kernels
        svm_kernels = kernels

    try:
        os.stat("tune_svr_logs")
    except:
        os.mkdir("tune_svr_logs")
    final_result_output = 'tune_svr_logs/{0}.tune.log'.format(train_file.split("/")[-1])
    fout = open(final_result_output, 'w')

    if scale:
        print "\tusing scale"
        scale_paras = ' -s {0}.scale.para {0} > {0}.scale'.format(train_file)
        execute(svm_scale + scale_paras)
        scale_paras = ' -r {0}.scale.para {1} > {1}.scale'.format(train_file, valid_file)
        execute((svm_scale + scale_paras))
        train_file = '{0}.scale'.format(train_file)
        valid_file = '{0}.scale'.format(valid_file)
    else:
        print "\ttraining without scale"

    best_model = {"mse": 100000., "scc": 1., "kernel": -1, "c": -1, "g": -1, 'eps': -1}

    for kernel in svm_kernels:
        model_file = valid_file + ".model"
        output_file = valid_file + ".output"
        if kernel == 2: # tune both c, g and epsilon
            for c in Cparas:
                for g in Gparas:
                    for eps in Epsilon:
                        # sys.stdout.write('Kernel:{0}, C paras:{1}, Gamma:{2}'.format(kernel_name[kernel], c, g))
                        fout.write('Kernel:{0}, C:{1}, Gamma:{2}, Eps:{3}\n'
                                   .format(kernel_name[kernel], c, g, eps))
                        train_paras = ' -s 3 -t {0} -c {1} -g {2} -p {3} -q {4} {5}'\
                                      .format(kernel, c, g, eps, train_file, model_file)
                        execute(liblinear_train + train_paras)
                        valid_paras = ' {0} {1} {2}'.format(valid_file, model_file, output_file)
                        result = execute(liblinear_test + valid_paras, False)
                        fout.write(result+"\n")
                        mse, scc = parse_result(result)
                        if mse < best_model["mse"]:
                            best_model["mse"] = mse
                            best_model["scc"] = scc
                            best_model["kernel"] = kernel
                            best_model["c"] = c
                            best_model["g"] = g
                            best_model["eps"] = eps
        else: # tune c and epsilon
            for c in Cparas:
                for eps in Epsilon:
                    # sys.stdout.write('Kernel:{0}, C paras:{1} '.format(kernel_name[kernel], c))
                    fout.write('Kernel:{0}, C:{1}, Eps:{2}\n'.format(kernel_name[kernel], c, eps))
                    train_paras = ' -s 3 -t {0} -c {1} -p {2} -q {3} {4}'.format(kernel, c, eps, train_file, model_file)
                    execute(liblinear_train + train_paras)
                    valid_paras = ' {0} {1} {2}'.format(valid_file, model_file, output_file)
                    result = execute(liblinear_test + valid_paras, False)
                    fout.write(result+"\n")
                    mse, scc = parse_result(result)
                    if mse < best_model["mse"]:
                        best_model["mse"] = mse
                        best_model["scc"] = scc
                        best_model["kernel"] = kernel
                        best_model["c"] = c
                        best_model["eps"] = eps
    print "\tBest model: Kernel:{0}, C:{1}, Gamma:{2}, Eps:{3}, MSE:{4}, SCC:{5}\n"\
           .format(best_model["kernel"], best_model["c"], best_model["g"], best_model["eps"],
                   best_model["mse"], best_model["scc"])
    fout.close()
