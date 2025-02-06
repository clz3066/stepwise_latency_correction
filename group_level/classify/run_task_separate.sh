#!/bin/bash
mkdir -p ./general_log
usr="linux"

if  [ $usr = "linux" ]; then
    echo "on linux platform"
    source activate priming_env
    CUDA_VISIBLE_DEVICES=1 nohup taskset -c 41-44 python -u classifier_task_separate.py "Fsp" "step0" > "./general_log/Fsp_step0.log" 2>&1 &
elif  [ $usr = "mac" ]; then
    echo "on mac platform"
    /usr/local/bin/python3 -u classifier_task_separate.py "Fsp" "step0" > "./general_log/Fsp_step0.log"
else
    echo  "wrong platform"
fi



