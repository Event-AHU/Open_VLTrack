LOG_FILE="/rydata/jinliye/RL/vltracking/EasyR1/log/$(date +'%Y-%m-%d_%H-%M-%S')_logfile_fulldataset_removeioureward.log"  # 添加日期前缀

bash examples/aqwen2_5_vl_3b_fulldataset_grpo.sh  >  ${LOG_FILE} 2>&1 &

