#!/bin/bash
mkdir -p ./mvpa_log
usr="linux"

if [ $usr = "windows" ]; then
    echo "on windows platform"
    dos2unix run_general.sh
    for data in "${data_name_lst[@]}"
    do
        echo $data
        CUDA_VISIBLE_DEVICES=0 C:/Users/20482981/Miniconda3/python -u classifier_general.py $data > "./general_log/"$data".log" 
    done
elif  [ $usr = "linux" ]; then
    echo "on linux platform"
    source activate priming_env
    # for data in "${data_name_lst[@]}"
    # do
        # echo $data
    task="Facc"
    gpu=0
    random_seed=5
    CUDA_VISIBLE_DEVICES=$gpu nohup taskset -c 11 python -u classifier_mvpa.py $task "s_component" $random_seed > "./mvpa_log/"$task"s_component_"$random_seed".log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$gpu nohup taskset -c 12 python -u classifier_mvpa.py $task "step0" $random_seed > "./mvpa_log/"$task"step0_"$random_seed".log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$gpu nohup taskset -c 13 python -u classifier_mvpa.py $task "step1" $random_seed > "./mvpa_log/"$task"step1_"$random_seed".log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=$gpu nohup taskset -c 44 python -u classifier_mvpa.py $task "step3" $random_seed > "./mvpa_log/"$task"step3_"$random_seed".log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=$gpu nohup taskset -c 45 python -u classifier_mvpa.py $task "step4" $random_seed > "./mvpa_log/"$task"step4_"$random_seed".log" 2>&1 &

    task="Fsp"
    gpu=0
    CUDA_VISIBLE_DEVICES=$gpu nohup taskset -c 16 python -u classifier_mvpa.py $task "s_component" $random_seed > "./mvpa_log/"$task"s_component_"$random_seed".log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$gpu nohup taskset -c 17 python -u classifier_mvpa.py $task "step0" $random_seed > "./mvpa_log/"$task"step0_"$random_seed".log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$gpu nohup taskset -c 18 python -u classifier_mvpa.py $task "step1" $random_seed > "./mvpa_log/"$task"step1_"$random_seed".log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=$gpu nohup taskset -c 49 python -u classifier_mvpa.py $task "step3" $random_seed > "./mvpa_log/"$task"step3_"$random_seed".log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=$gpu nohup taskset -c 50 python -u classifier_mvpa.py $task "step4" $random_seed > "./mvpa_log/"$task"step4_"$random_seed".log" 2>&1 &
elif  [ $usr = "mac" ]; then
    echo "on mac platform"
    nohup /usr/local/bin/python3 -u classifier_mvpa.py "Facc" "step0" > "./mvpa_log/Faccstep0.log" 2>&1 &
else
    echo  "wrong platform"
fi



