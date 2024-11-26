rm -r cache_dir
python finetune.py --do_eval --eval_dataset low1 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --eval_dataset low2 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --eval_dataset low3 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --eval_dataset low4 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --eval_dataset low5 --model_type albert --model_name_or_path albert/albert-base-v2

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset low1 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset low2 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset low3 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset low4 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset low5 --model_type albert --model_name_or_path albert/albert-base-v2

python finetune.py --do_eval --train_dataset rem-80 --eval_dataset low1 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset low2 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset low3 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset low4 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset low5 --model_type albert --model_name_or_path albert/albert-base-v2

python finetune.py --do_eval --train_dataset rem-60 --eval_dataset low1 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset low2 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset low3 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset low4 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset low5 --model_type albert --model_name_or_path albert/albert-base-v2

python finetune.py --do_eval --train_dataset rem-40 --eval_dataset low1 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset low2 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset low3 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset low4 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset low5 --model_type albert --model_name_or_path albert/albert-base-v2

python finetune.py --do_eval --train_dataset rem-20 --eval_dataset low1 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset low2 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset low3 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset low4 --model_type albert --model_name_or_path albert/albert-base-v2
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset low5 --model_type albert --model_name_or_path albert/albert-base-v2