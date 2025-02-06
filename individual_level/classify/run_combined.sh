#!/bin/bash
mkdir -p ./combined_log
usr="linux"

if  [ $usr = "linux" ]; then
    echo "on linux platform"
    source activate priming_env
    random_seed="1"
    CUDA_VISIBLE_DEVICES=0 nohup taskset -c 1-4 python -u classifier_combined.py "step0" $random_seed > "./combined_log/step0_"$random_seed".log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=2 nohup taskset -c 5-8 python -u classifier_combined.py "step1" $random_seed > "./combined_log/step1_"$random_seed".log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=0 nohup taskset -c 11-14 python -u classifier_combined.py "step2" $random_seed > "./combined_log/step2_"$random_seed".log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=0 nohup taskset -c 15-18 python -u classifier_combined.py "step3" $random_seed > "./combined_log/step3_"$random_seed".log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=1 nohup taskset -c 11-14 python -u classifier_combined.py "step4" $random_seed > "./combined_log/step4_"$random_seed".log" 2>&1 &
elif  [ $usr = "mac" ]; then
    /usr/local/bin/python3 -u classifier_subject.py "Facc" "step4" > "./subject_log/Fsp.log"
else
    echo  "wrong platform"
fi




