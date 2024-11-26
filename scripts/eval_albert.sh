rm -r cache_dir
python finetune.py --do_eval --model_type albert --model_name_or_path albert/albert-base-v2 --output_dir experiments --eval_dataset rem1
python finetune.py --do_eval --model_type albert --model_name_or_path albert/albert-base-v2 --output_dir experiments --eval_dataset rem2
python finetune.py --do_eval --model_type albert --model_name_or_path albert/albert-base-v2 --output_dir experiments --eval_dataset rem3
python finetune.py --do_eval --model_type albert --model_name_or_path albert/albert-base-v2 --output_dir experiments --eval_dataset rem4
python finetune.py --do_eval --model_type albert --model_name_or_path albert/albert-base-v2 --output_dir experiments --eval_dataset rem5

rm -r cache_dir
python finetune.py --do_eval --train_dataset rem-100 --model_type albert --model_name_or_path albert/albert-base-v2 --output_dir experiments --eval_dataset rem1
python finetune.py --do_eval --train_dataset rem-100 --model_type albert --model_name_or_path albert/albert-base-v2 --output_dir experiments --eval_dataset rem2
python finetune.py --do_eval --train_dataset rem-100 --model_type albert --model_name_or_path albert/albert-base-v2 --output_dir experiments --eval_dataset rem3
python finetune.py --do_eval --train_dataset rem-100 --model_type albert --model_name_or_path albert/albert-base-v2 --output_dir experiments --eval_dataset rem4
python finetune.py --do_eval --train_dataset rem-100 --model_type albert --model_name_or_path albert/albert-base-v2 --output_dir experiments --eval_dataset rem5