MODE=$1

# Dataset config
ROOT=data/DWC
GPU=2
CFG=config/cfg_odvg.py
DATASET=config/diverse_weather.json

# Model config
BATCH_SIZE=8
EPOCHS=1

if [ "$MODE" == "debug" ]; then
    SEEDS=(1)
else
    SEEDS=(1)
fi


NNODES=${NNODES:-1} 
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# bash script/train.sh zeroshot

for SEED in "${SEEDS[@]}"
do          
    OUTPUT_DIR=work_dirs/${MODE}/seed${SEED}_bs4_ep${EPOCHS}
    PRETRAINER_MODEL_PATH=data/weights/gdinot-1.8m-odvg.pth
    echo "Run this job and save the output to ${DIR}"
    python -m torch.distributed.launch  --nproc_per_node=${GPU} main.py \
            --output_dir ${OUTPUT_DIR} \
            --seed ${SEED} \
            -c ${CFG} \
            --datasets ${DATASET}  \
            --pretrain_model_path ${PRETRAINER_MODEL_PATH} \
            --options text_encoder_type=./data/weights/bert_base_uncased batch_size=${BATCH_SIZE} epochs=${EPOCHS}
done