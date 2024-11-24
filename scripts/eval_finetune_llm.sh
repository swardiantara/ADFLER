rm -r cache_dir
python finetune.py --do_eval --train_dataset rem-100 --model_type roberta --model_name_or_path roberta-base --output_dir experiments --eval_dataset rem1
python finetune.py --do_eval --train_dataset rem-100 --model_type roberta --model_name_or_path roberta-base --output_dir experiments --eval_dataset rem2
python finetune.py --do_eval --train_dataset rem-100 --model_type roberta --model_name_or_path roberta-base --output_dir experiments --eval_dataset rem3
python finetune.py --do_eval --train_dataset rem-100 --model_type roberta --model_name_or_path roberta-base --output_dir experiments --eval_dataset rem4
python finetune.py --do_eval --train_dataset rem-100 --model_type roberta --model_name_or_path roberta-base --output_dir experiments --eval_dataset rem5