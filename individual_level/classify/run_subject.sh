#!/bin/bash
mkdir -p ./subject_log
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
    CUDA_VISIBLE_DEVICES=1 nohup taskset -c 51-54 python -u classifier_subject.py "Facc" "s_component" "2" > "./subject_log/Facc_s_component_2.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 nohup taskset -c 55-58 python -u classifier_subject.py "Fsp"  "s_component" "2" > "./subject_log/Fsp_s_component_2.log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=2 nohup taskset -c 1-4 python -u classifier_subject.py "Facc" "r_component" "2" > "./subject_log/Facc_r_component_2.log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=2 nohup taskset -c 5-8 python -u classifier_subject.py "Fsp"  "r_component" "2" > "./subject_log/Fsp_r_component_2.log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=0 nohup taskset -c 31-34 python -u classifier_subject.py "Facc" "step3" "2" > "./subject_log/Facc_step3_2.log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=0 nohup taskset -c 35-38 python -u classifier_subject.py "Fsp" "step3" "2" > "./subject_log/Fsp_step3_2.log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=0 nohup taskset -c 41-44 python -u classifier_subject.py "Facc" "step4" "2" > "./subject_log/Facc_step4_2.log" 2>&1 &
    # CUDA_VISIBLE_DEVICES=0 nohup taskset -c 45-48 python -u classifier_subject.py "Fsp" "step4" "2" > "./subject_log/Fsp_step4_2.log" 2>&1 &
elif  [ $usr = "mac" ]; then
    /usr/local/bin/python3 -u classifier_subject.py "Facc" "step4" > "./subject_log/Fsp.log"
else
    echo  "wrong platform"
fi



