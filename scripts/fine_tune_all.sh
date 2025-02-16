# bash/sh
train_sets=( rem-100 )
eval_sets=( ori1 ori2 ori3 ori4 ori5 rem1 rem2 rem3 rem4 rem5 )
seeds=(17 42 67 99 123)


# BERT-base-cased
for train_set in "${train_sets[@]}"; do
    for seed in "${seeds[@]}"; do
# train original
    # test DT1 -> original
        rm -r cache_dir
        python finetune.py --do_train --train_dataset "$train_set" --do_eval --seed "$seed" --model_type bert --model_name_or_path bert-base-uncased --scenario capital
        # test DT2 -> ori1 s.d ori5
        # test DT3 -> rem1 s.d rem5
        for eval_set in "${eval_sets[@]}"; do
            python finetune.py --train_dataset "$train_set" --do_eval --eval_dataset "$eval_set" --seed "$seed" --model_type bert --model_name_or_path bert-base-uncased --scenario capital
        done
    done
done

# # RoBERTa-base
# for train_set in "${train_sets[@]}"; do
#     for seed in "${seeds[@]}"; do
# # train original
#     # test DT1 -> original
#         rm -r cache_dir
#         python finetune.py --do_train --train_dataset "$train_set" --do_eval --seed "$seed" --model_type roberta --model_name_or_path roberta-base --scenario pc-ajk
#         # test DT2 -> ori1 s.d ori5
#         # test DT3 -> rem1 s.d rem5
#         for eval_set in "${eval_sets[@]}"; do
#             python finetune.py --train_dataset "$train_set" --do_eval --eval_dataset "$eval_set" --seed "$seed" --model_type roberta --model_name_or_path roberta-base --scenario pc-ajk
#         done
#     done
# done

# # DistilBERT-base-cased
# for train_set in "${train_sets[@]}"; do
#     for seed in "${seeds[@]}"; do
# # train original
#     # test DT1 -> original
#         rm -r cache_dir
#         python finetune.py --do_train --train_dataset "$train_set" --do_eval --seed "$seed" --model_type distilbert --model_name_or_path distilbert-base-cased --scenario pc-ajk
#         # test DT2 -> ori1 s.d ori5
#         # test DT3 -> rem1 s.d rem5
#         for eval_set in "${eval_sets[@]}"; do
#             python finetune.py --train_dataset "$train_set" --do_eval --eval_dataset "$eval_set" --seed "$seed" --model_type distilbert --model_name_or_path distilbert-base-cased --scenario pc-ajk
#         done
#     done
# done

# # DistilRoBERTa-base
# for train_set in "${train_sets[@]}"; do
#     for seed in "${seeds[@]}"; do
# # train original
#     # test DT1 -> original
#         rm -r cache_dir
#         python finetune.py --do_train --train_dataset "$train_set" --do_eval --seed "$seed" --model_type roberta --model_name_or_path distilroberta-base --scenario pc-ajk
#         # test DT2 -> ori1 s.d ori5
#         # test DT3 -> rem1 s.d rem5
#         for eval_set in "${eval_sets[@]}"; do
#             python finetune.py --train_dataset "$train_set" --do_eval --eval_dataset "$eval_set" --seed "$seed" --model_type roberta --model_name_or_path distilroberta-base --scenario pc-ajk
#         done
#     done
# done

# # XLNet-base-cased
# for train_set in "${train_sets[@]}"; do
#     for seed in "${seeds[@]}"; do
# # train original
#     # test DT1 -> original
#         rm -r cache_dir
#         python finetune.py --do_train --train_dataset "$train_set" --do_eval --seed "$seed" --model_type xlnet --model_name_or_path xlnet-base-cased --scenario pc-ajk
#         # test DT2 -> ori1 s.d ori5
#         # test DT3 -> rem1 s.d rem5
#         for eval_set in "${eval_sets[@]}"; do
#             python finetune.py --train_dataset "$train_set" --do_eval --eval_dataset "$eval_set" --seed "$seed" --model_type xlnet --model_name_or_path xlnet-base-cased --scenario pc-ajk
#         done
#     done
# done

# # ALBERT-base-V2
# for train_set in "${train_sets[@]}"; do
#     for seed in "${seeds[@]}"; do
# # train original
#     # test DT1 -> original
#         rm -r cache_dir
#         python finetune.py --do_train --train_dataset "$train_set" --do_eval --seed "$seed" --model_type albert --model_name_or_path albert/albert-base-v2 --scenario pc-ajk
#         # test DT2 -> ori1 s.d ori5
#         # test DT3 -> rem1 s.d rem5
#         for eval_set in "${eval_sets[@]}"; do
#             python finetune.py --train_dataset "$train_set" --do_eval --eval_dataset "$eval_set" --seed "$seed" --model_type albert --model_name_or_path albert/albert-base-v2 --scenario pc-ajk
#         done
#     done
# done

# # Electra-base
# for train_set in "${train_sets[@]}"; do
#     for seed in "${seeds[@]}"; do
# # train original
#     # test DT1 -> original
#         rm -r cache_dir
#         python finetune.py --do_train --train_dataset "$train_set" --do_eval --seed "$seed" --model_type electra --model_name_or_path google/electra-base-discriminator --scenario pc-ajk
#         # test DT2 -> ori1 s.d ori5
#         # test DT3 -> rem1 s.d rem5
#         for eval_set in "${eval_sets[@]}"; do
#             python finetune.py --train_dataset "$train_set" --do_eval --eval_dataset "$eval_set" --seed "$seed" --model_type electra --model_name_or_path google/electra-base-discriminator --scenario pc-ajk
#         done
#     done
# done