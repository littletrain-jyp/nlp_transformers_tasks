CUDA_VISIBLE_DEVICES='0,1'
set -x
export NOWTIME=$(date +%Y-%m-%d_%H-%M-%S)
export TASKNAME="text_classification"
export OUTPUT_DIR=./runs/${TASKNAME}_${NOWTIME}
export OUTPUT_LOG_DIR==./runs/${TASKNAME}_${NOWTIME}/logs
#export OUTPUT_CKPT_DIR==./runs/${TASKNAME}_${NOWTIME}/checkpoints/

mkdir -p ./runs
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_LOG_DIR}
#mkdir -p ${OUTPUT_CKPT_DIR}

export CONFIG_PATH="./conf/config.yml"

python main.py \
--do_train \
--output_dir $OUTPUT_DIR\
--output_log_dir $OUTPUT_LOG_DIR\
--config $CONFIG_PATH

