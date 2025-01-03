# Evaluate BERT-base-cased trained on original dataset

# Evaluate on original sensitivity test (DT2)
python finetune.py --do_eval --eval_dataset original --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --eval_dataset original --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --eval_dataset original --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --eval_dataset original --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --eval_dataset original --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67

# Evaluate on punctuation-removed sensitivity test (DT3)
python finetune.py --do_eval --eval_dataset rem-100 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --eval_dataset rem-100 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --eval_dataset rem-100 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --eval_dataset rem-100 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --eval_dataset rem-100 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67