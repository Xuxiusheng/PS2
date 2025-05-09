MODE=$1
# Dataset config
ROOT=data/DWC
GPU=2
CFG=config/cfg_odvg.py
DATASET=config/diverse_weather.json
ALL_DOMAIN=("Daytime_Sunny" "Daytime_Foggy" "Dusk_Rainy" "Night_Rainy" "Night_Sunny")
# ALL_DOMAIN=("Daytime_Foggy")
SEEDS=(1)
# ALL_DOMAIN=("All")

# Model config
BATCH_SIZE=4

# GPU_NUM=$1
# CFG=$2
# DATASETS=$3
# OUTPUT_DIR=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# bash script/test.sh zs-dwc

for SEED in "${SEEDS[@]}"
do
    PRETRAINER_MODEL_PATH=work_dirs/train-b02-urban/seed${SEED}_bs4_ep1/checkpoint_best_regular.pth
    for DOMAIN in "${ALL_DOMAIN[@]}"
    do 
        echo "Run ${DOMAIN} with seed ${SEED}"
        python tools/d2d.py -j ${DATASET} -f "root anno" -v "${ROOT}/${DOMAIN}/val ${ROOT}/${DOMAIN}/annotations/val.json"
        
        OUTPUT_DIR=work_dirs/${MODE}/seed${SEED}/${DOMAIN}/bs${BATCH_SIZE}

        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}, so skip this job"
        else
            echo "Run this job and save the output to ${DIR}"
            python -m torch.distributed.launch  --nproc_per_node=${GPU} main.py \
                    --output_dir ${OUTPUT_DIR} \
                    --eval \
                    --fix_size \
                    -c ${CFG} \
                    --datasets ${DATASET}  \
                    --pretrain_model_path ${PRETRAINER_MODEL_PATH} \
                    --options text_encoder_type=./data/weights/bert_base_uncased batch_size=${BATCH_SIZE}
        fi
    done
done