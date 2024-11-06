python bert_ner.py \
   --model_name_or_path bert-base-cased \
   --do_train \
   --output_dir development \
   --train_batch_size 16 \
   --eval_batch_size 16 \
   --learning_rate 2e-5 \
   --max_seq_length 128 \
   --train_epochs 5 \
   --seed 42 \
#    --align_label \
#    --save_model \