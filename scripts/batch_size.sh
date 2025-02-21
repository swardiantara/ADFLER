# bash/sh
train_sets=( original )
# eval_sets=( ori1 ori2 ori3 ori4 ori5 rem1 rem2 rem3 rem4 rem5 low1 low2 low3 low4 low5)
seeds=(17 42 67 99 123)
batch_sizes=(4 8 16 32 64 128)

# BERT-base-uncased
for train_set in "${train_sets[@]}"; do
    for seed in "${seeds[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
    # train original
        # test DT1 -> original
            rm -r cache_dir
            python finetune.py --do_train --train_dataset "$train_set" --do_eval --seed "$seed" --model_type bert --model_name_or_path bert-base-uncased --scenario batch_size --train_batch_size "$batch_size" --eval_batch_size "$batch_size"
            # python interpret_predictions.py --train_dataset "$train_set" --seed "$seed" --model_type bert --model_name_or_path bert-base-uncased --scenario final
            # test DT2 -> ori1 s.d ori5
            # test DT3 -> rem1 s.d rem5
            # test DT4 -> low1 s.d low5
            # for eval_set in "${eval_sets[@]}"; do
            #     python finetune.py --train_dataset "$train_set" --do_eval --eval_dataset "$eval_set" --seed "$seed" --model_type bert --model_name_or_path bert-base-uncased --scenario final
            # done
        done
    done
done