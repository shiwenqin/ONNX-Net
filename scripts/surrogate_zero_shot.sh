seed=42

accelerate launch --num_processes 4 ../src/bert_tuning.py \
        --model_name Qwen/Qwen3-1.7B \
        --data_path ../chain_slim_v1/nb101/nb101_50000_$seed.csv \
        --eval_path ../chain_slim_v1/nasbench201.csv \
        --output_path ../res/ \
        --batch_size 16 \
        --epochs 5 \
        --seed $seed \
        --lr 5e-5 \
        --loss_fn pwr \
        --weight_decay 0.1 \
        --eval_strategy epoch \
        --gradient_checkpointing True