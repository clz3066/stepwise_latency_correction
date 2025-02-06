#!/bin/bash
mkdir -p ./combined_log
usr="linux"

if  [ $usr = "linux" ]; then
    echo "on linux platform"
    source activate priming_env
    CUDA_VISIBLE_DEVICES=0 nohup taskset -c 11-14 python -u classifier_task_combined.py "step0"  > "./combined_log/step0.log" 2>&1 &
elif  [ $usr = "mac" ]; then
    /usr/local/bin/python3 -u classifier_task_combined.py "step0" > "./combined_log/step0.log"
else
    echo  "wrong platform"
fi




