#!/usr/bin/env bash

tune_log=$1
 [ -d "results/" ] || mkdir -p "results/"

while IFS= read -r line
do
    IFS= read -r path
    if [ ! $path ]
    then
        break
    fi
    IFS='/' read -r -a folders <<< $path
    folder_feature=${folders[-2]}
    folder_base=${folders[-1]}

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

    echo "training for" $folder_base, kernel:$kernel, c:$c, g:$g

    obj_path="results/"$folder_feature/$folder_base
     [ -d $obj_path ] || mkdir -p $obj_path

    train_file=$path/$folder_base".train"
    test_file=$path/$folder_base".test"
    model="$obj_path/model.$folder_base"
    output="$obj_path/output.$folder_base"

    ./svm-scale -s "$obj_path/train.scale.para" "$train_file" > "$train_file.scaled"
    ./svm-scale -r "$obj_path/train.scale.para" "$test_file" > "$test_file.scaled"
    ./svm-train -s 0 -t $kernel -c $c -g $g -q "$train_file.scaled" "$model"
    ./svm-predict "$test_file.scaled" "$model" "$output"

    echo ""

done < "$tune_log"