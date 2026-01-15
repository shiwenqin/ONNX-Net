SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

accelerate launch --num_processes 4 $SCRIPT_DIR/../src/onnxnet/bert_tuning.py \
        --model_name answerdotai/ModernBERT-large \
        --data_path $SCRIPT_DIR/../data/chain_slim_v1/ \
        --eval_task nas201nats \
        --output_path $SCRIPT_DIR/../res/ \
        --batch_size 16 \
        --epochs 5 \
        --seed 42 \
        --lr 5e-5 \
        --loss_fn pwr \
        --weight_decay 0.1 \
        --eval_strategy epoch \
        --gradient_checkpointing True