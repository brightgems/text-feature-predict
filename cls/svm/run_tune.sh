#!/usr/bin/env bash

kernel=2 # use rbf kernel (for topic modeling)
# kernel=0 # use linear kernel (for bow/tfidf baseline)

path_in="/home/yiren/Documents/time-series-predict/data/features_reg"
echo $path_in

tune_model=$1
if [ $tune_model = 'svr' ];
then
    tune_script="tune_params_svr.py"
    echo "tuning svr (epsilon-svm)"
else
    tune_script="tune_params_svm.py"
    echo "tuning svm (c-svm)"
fi

tune_folder=$2
IFS=';' read -ra folders <<< $tune_folder

for path_feature in $(find $path_in/* -maxdepth 0 -mindepth 0 -type d);
do
    if [ $tune_folder ];
    then
        flag=0
        for folder in "${folders[@]}";
        do
            # echo -e " \t $folder "
            if [ $folder = $(basename $path_feature) ];
            then
                flag=1
                break
            fi
        done
        if [ $flag = 0 ]
        then
            continue
        fi
    fi
    for folder in $(find $path_feature/* -maxdepth 1 -mindepth 0 -type d);
    do
        echo $folder;
        train_file=$folder/$(basename $folder)".train"
        valid_file=$folder/$(basename $folder)".valid"
        python $tune_script $train_file $valid_file 1 $kernel
        rm -f "$train_file.scale.para" "$train_file.scale" "$valid_file.scale" "$valid_file.scale.model" "$valid_file.scale.output"
    done
done


# ./run_tune.sh "svr" "topic_change;topic_hist_d0.7_w1;topic_hist_d0.7_w1_cont;topic_hist_d0.7_w2;topic_hist_d0.7_w2_cont;topic_hist_d0.7_w3;topic_hist_d0.7_w3_cont;topic_hist_d0.7_w4;topic_hist_d0.7_w4_cont;topic_hist_d0.7_w5" > tune_summary/log.reg.tune.1
# ./run_tune.sh "svr" "topic_hist_d0.7_w5_cont;topic_hist_d0.7_w6;topic_hist_d0.7_w6_cont;topic_hist_d0.8_w1;topic_hist_d0.8_w1_cont;topic_hist_d0.8_w2;topic_hist_d0.8_w2_cont;topic_hist_d0.8_w3;topic_hist_d0.8_w3_cont;topic_hist_d0.8_w4" > tune_summary/log.reg.tune.2
# ./run_tune.sh "svr" "topic_hist_d0.8_w4_cont;topic_hist_d0.8_w5;topic_hist_d0.8_w5_cont;topic_hist_d0.8_w6;topic_hist_d0.8_w6_cont;topic_hist_d0.9_w1;topic_hist_d0.9_w1_cont;topic_hist_d0.9_w2;topic_hist_d0.9_w2_cont;topic_hist_d0.9_w3" > tune_summary/log.reg.tune.3
# ./run_tune.sh "svr" "topic_hist_d0.9_w3_cont;topic_hist_d0.9_w4;topic_hist_d0.9_w4_cont;topic_hist_d0.9_w5;topic_hist_d0.9_w5_cont;topic_hist_d0.9_w6;topic_hist_d0.9_w6_cont;topic_hist_d1_w1;topic_hist_d1_w1_cont;topic_hist_d1_w2" > tune_summary/log.reg.tune.4
# ./run_tune.sh "svr" "topic_hist_d1_w2_cont;topic_hist_d1_w3;topic_hist_d1_w3_cont;topic_hist_d1_w4;topic_hist_d1_w4_cont;topic_hist_d1_w5;topic_hist_d1_w5_cont;topic_hist_d1_w6;topic_hist_d1_w6_cont" > tune_summary/log.reg.tune.5

