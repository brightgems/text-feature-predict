#!/usr/bin/env bash

kernel=2 # use rbf kernel (for topic modeling)
# kernel=0 # use linear kernel (for bow/tfidf baseline)

tune_folder=$1
echo $tune_folder
IFS=';' read -ra folders <<< $tune_folder

path_in="/home/yiren/Documents/time-series-predict/data/test"
echo $path_in

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
        python tune_params_svm.py $train_file $valid_file 1 $kernel
        rm -f "$train_file.scale.para" "$train_file.scale" "$valid_file.scale" "$valid_file.scale.model" "$valid_file.scale.output"
    done
done


