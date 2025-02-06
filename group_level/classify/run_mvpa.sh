#!/bin/bash
mkdir -p mvpa_log
usr="linux"
k_fold="0"

if  [ $usr = "linux" ]; then
    echo "on linux platform"
    source activate priming_env
    CUDA_VISIBLE_DEVICES=1 nohup taskset -c 18 python -u classifier_mvpa.py "Fsp" "step0" $k_fold > "./mvpa_log/Fsp_step0_"$k_fold".log" 2>&1 &
elif  [ $usr = "mac" ]; then
    /usr/local/bin/python3 -u classifier_mvpa.py "Fsp" "step0" $k_fold > "./mvpa_log/Fsp_step0_"$k_fold".log"
else
    echo  "wrong platform"
fi

