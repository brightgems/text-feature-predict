#!/usr/bin/env bash

tune_log=$1
path="/home/yiren/Documents/time-series-predict/data/test/"
 [ -d "results/" ] || mkdir -p "results/"

while IFS= read -r line
do
    IFS= read -r feature
    if [ ! $feature ]
    then
        break
    fi
    IFS= read -r line
    IFS= read -r line
    IFS= read -r model

    # read best params from tuning log
    IFS=', ' read -r -a params <<< $model
    IFS=':' read -r -a kernel <<< ${params[2]}
    kernel=${kernel[1]}
    IFS=':' read -r -a c <<< ${params[3]}
    c=${c[1]}
    IFS=':' read -r -a g <<< ${params[4]}
    g=${g[1]}

    echo "training for" $feature, kernel:$kernel, c:$c, g:$g

    # todo: may need parsing here
    upper_folder=$feature

    train_path=$path/$upper_folder/$feature/
    obj_path="results/"$upper_folder/$feature
     [ -d $obj_path ] || mkdir -p $obj_path

    train_file=$train_path/$feature".train"
    test_file=$train_path/$feature".test"
    model="$obj_path/model.$feature"
    output="$obj_path/output.$feature"

    ./svm-scale -s "$obj_path/train.scale.para" "$train_file" > "$train_file.scaled"
    ./svm-scale -r "$obj_path/train.scale.para" "$test_file" > "$test_file.scaled"
    ./svm-train -s 0 -t $kernel -c $c -g $g -q "$train_file.scaled" "$model"
    ./svm-predict "$test_file.scaled" "$model" "$output"

    echo ""

done < "$tune_log"