# Original Dataset for Train and Test
rm -r cache_dir
python finetune.py --do_train --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
rm -r cache_dir
python finetune.py --do_train --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
rm -r cache_dir
python finetune.py --do_train --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
rm -r cache_dir
python finetune.py --do_train --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
rm -r cache_dir
python finetune.py --do_train --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67

rm -r cache_dir
python finetune.py --do_train --do_eval --model_type bert --model_name_or_path bert-base-uncased --scenario capital --seed 42
rm -r cache_dir
python finetune.py --do_train --do_eval --model_type bert --model_name_or_path bert-base-uncased --scenario capital --seed 123
rm -r cache_dir
python finetune.py --do_train --do_eval --model_type bert --model_name_or_path bert-base-uncased --scenario capital --seed 99
rm -r cache_dir
python finetune.py --do_train --do_eval --model_type bert --model_name_or_path bert-base-uncased --scenario capital --seed 17
rm -r cache_dir
python finetune.py --do_train --do_eval --model_type bert --model_name_or_path bert-base-uncased --scenario capital --seed 67