#!/bin/bash
mkdir -p ./rnn_log
usr="linux"

echo "on linux platform"
source activate priming_env
task="Fsp"
CUDA_VISIBLE_DEVICES=0 nohup taskset -c 21-24 python -u classifier_rnn.py $task "s_component" "0" > "./rnn_log/"$task"s_component_0.log" 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup taskset -c 5-8 python -u classifier_rnn.py $task "s_component" "1" > "./rnn_log/"$task"s_component_1.log" 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup taskset -c 15-18 python -u classifier_rnn.py $task "s_component" "2" > "./rnn_log/"$task"s_component_2.log" 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup taskset -c 15-18 python -u classifier_rnn.py $task "step3" "0" > "./rnn_log/"$task"step3_0.log" 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup taskset -c 5-8 python -u classifier_rnn.py $task "step4" "0" > "./rnn_log/"$task"step4_0.log" 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup taskset -c 5-8 python -u classifier_rnn.py $task "step4" "2" > "./rnn_log/"$task"step4_2.log" 2>&1 &




