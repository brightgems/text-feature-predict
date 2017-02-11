"""
batch tune for cross-validation with multi-sampling
previous
"""

from tune_params_svm import *

import os

if __name__ == "__main__":

    st = 'topic_combined_sentiment'
    path_data = '../../../data/features/cross_validation/' + st + '/'
    path_result = 'tune/' + st +'/'
    num_folder = 5

    svm_train = './svm-train'
    svm_scale = './svm-scale'
    svm_predict = './svm-predict'


    folders = os.listdir(path_data)
    try:
        os.stat(path_result)
    except:
        os.mkdir(path_result)

    for folder in folders:

        #if "15" not in folder:
            #continue

        print "tuning", folder
        valid_file = path_data + folder + "/" + folder + ".valid"
        f_log = open(path_result+'log.{0}'.format(folder), "w")

        for k in range(num_folder):
            train_file = path_data + folder + "/{0}/{1}.train.s{2}.k0".format(k, folder, k)
            valid_file = path_data + folder + "/" + folder + ".valid"
            print train_file
            print "tune on {1}.train.s{0}.k0".format(k, folder)
            f_log.write("\n=======================\ntune on {1}.train.s{0}.k0"
                        "\n=======================\n".format(k, folder))

            # scale
            scale_paras = ' -s {1}train.scale.para {0} > {1}train.scale'.format(train_file, path_result)
            execute(svm_scale + scale_paras)
            scale_paras = ' -r {1}train.scale.para {0} > {1}valid.scale'.format(valid_file, path_result)
            execute((svm_scale + scale_paras))
            train_file = path_result+'train.scale'
            valid_file = path_result+'valid.scale'

            # tune
            for kernel in kernels:
                model_file = path_result+kernel_name[kernel] + '.model.tune'
                for i in xrange(len(Cparas)):
                    print '{0} Kernel, C paras:{1}\t'.format(kernel_name[kernel], Cparas[i])
                    f_log.write('{0} Kernel, C paras:{1}\t'.format(kernel_name[kernel], Cparas[i]))
                    train_paras = ' -s 0 -t {0} -c {1} -q {2} {3}'.format(kernel, Cparas[i], train_file, model_file)
                    execute(svm_train + train_paras)
                    valid_paras = ' {0} {1} {2}valid_predict.output.txt'.format(valid_file, model_file, path_result)
                    result = execute(svm_predict + valid_paras, True)
                    f_log.write(result)
        f_log.close()



