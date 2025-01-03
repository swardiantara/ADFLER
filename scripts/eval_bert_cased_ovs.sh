# Evaluate BERT-base-cased trained on original dataset

# Evaluate on original sensitivity test (DT2)
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori1 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori1 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori1 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori1 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori1 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori2 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori2 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori2 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori2 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori2 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori3 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori3 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori3 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori3 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori3 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori4 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori4 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori4 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori4 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori4 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori5 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori5 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori5 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori5 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori5 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67

# Evaluate on punctuation-removed sensitivity test (DT3)
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem1 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem1 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem1 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem1 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem1 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem2 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem2 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem2 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem2 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem2 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem3 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem3 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem3 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem3 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem3 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem4 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem4 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem4 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem4 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem4 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem5 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 42
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem5 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 123
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem5 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 99
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem5 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 17
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem5 --model_type bert --model_name_or_path bert-base-cased --scenario capital --seed 67