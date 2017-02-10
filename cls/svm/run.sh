#!/bin/sh

classifer() {
    ststart="topic_dist"
    kernel=2 # ['linear', 'polynomial', 'rbf', 'sigmoid']
    c=1024
    #t=15

    st=$1
    stend=$2
    t=$3

    path="../../../data/features/cross_validation"
    filename="${ststart}_${t}${stend}"
    data_path="$path/${st}/$filename"
    echo $data_path
    obj_path="results/${st}.t${t}"

    [ -d $obj_path ] || mkdir -p $obj_path

    train_scaled="$obj_path/train.scaled"
    test_scaled="$obj_path/test.scaled"

    echo "C-SVM with kernel $kernel and C=$c (tuned on dev set)"

    for s in 0 1 2 3 4
    do
        echo ""
        for k in 0 1 2 3 4
        do
            train_file="$data_path/$s/$filename.train.s$s.k$k"
            test_file="$data_path/$s/$filename.test.s$s.k$k"
            model="$obj_path/model.t$t.s$s.k$k"
            output="$obj_path/output.t$t.s$s.k$k"

            #echo "cross validation on partition $idx"
            ./svm-scale -s "$obj_path/train.scale.para" "$train_file" > "$train_scaled"
            ./svm-scale -r "$obj_path/train.scale.para" "$test_file" > "$test_scaled"
            ./svm-train -s 0 -t $kernel -c $c -q "$train_scaled" "$model"
            ./svm-predict "$test_scaled" "$model" "$output.txt"
        done
    done

    echo ""
}


for t in 10 15 20 25 30 35 40 45 50
do 
    echo "topic number: $t"
    classifer "topic_dist" "" $t
    classifer "topic_sentiment" "_sentiment" $t
    classifer "topic_change" "_change" $t
    classifer "topic_change_sentiment" "_change_sentiment" $t
    classifer "topic_combined" "_combined" $t
    classifer "topic_combined_sentiment" "_combined_sentiment" $t
    echo ""
done