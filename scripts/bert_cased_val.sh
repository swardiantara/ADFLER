# Run BERT-base-cased on a train-test split by using seed-42 on 5 different seeds
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-42 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 42
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-42 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 123
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-42 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 99
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-42 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 17
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-42 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 67

# Run BERT-base-cased on a train-test split by using seed-123 on 5 different seeds
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-123 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 42
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-123 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 123
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-123 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 99
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-123 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 17
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-123 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 67

# Run BERT-base-cased on a train-test split by using seed-99 on 5 different seeds
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-99 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 42
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-99 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 123
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-99 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 99
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-99 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 17
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-99 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 67

# Run BERT-base-cased on a train-test split by using seed-17 on 5 different seeds
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-17 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 42
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-17 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 123
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-17 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 99
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-17 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 17
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-17 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 67

# Run BERT-base-cased on a train-test split by using seed-67 on 5 different seeds
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-67 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 42
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-67 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 123
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-67 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 99
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-67 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 17
rm -r cache_dir
python finetune.py --do_train --train_dataset seed-67 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario data-seed --seed 67
