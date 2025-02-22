python with_encoder.py \
    --model_type albert \
   --model_name_or_path albert/albert-base-v2 \
   --encoder lstm \
   --output_dir experiments \
   --train_batch_size 16 \
   --eval_batch_size 16 \
   --learning_rate 2e-5 \
   --max_seq_length 128 \
   --train_epochs 15 \
   --bidirectional \
   --num_layers 2 \
   --seed 42 \