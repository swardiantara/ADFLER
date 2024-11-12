rm -r cache_dir
python finetune.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --output_dir experiments \
    --train_epochs 15 \
    --learning_rate 5e-5 \