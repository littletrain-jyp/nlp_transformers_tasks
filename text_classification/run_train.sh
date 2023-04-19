set -x
export NOWTIME=$(date +%Y-%m-%d_%H-%M-%S)
export TASKNAME="text_classification"
export OUTPUT_DIR=./runs/${TASKNAME}_${NOWTIME}
export OUTPUT_LOG_DIR==./runs/${TASKNAME}_${NOWTIME}/logs
export OUTPUT_CKPT_DIR==./runs/${TASKNAME}_${NOWTIME}/checkpoints/

mkdir -p ./runs
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_LOG_DIR}
mkdir -p ${OUTPUT_CKPT_DIR}


