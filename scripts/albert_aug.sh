rm -r cache_dir
python finetune.py --do_train --do_eval --model_type albert --model_name_or_path albert/albert-base-v2

rm -r cache_dir
python finetune.py --do_train --train_dataset rem-100 --do_eval --model_type albert --model_name_or_path albert/albert-base-v2

rm -r cache_dir
python finetune.py --do_train --train_dataset rem-80 --do_eval --model_type albert --model_name_or_path albert/albert-base-v2

rm -r cache_dir
python finetune.py --do_train --train_dataset rem-60 --do_eval --model_type albert --model_name_or_path albert/albert-base-v2

rm -r cache_dir
python finetune.py --do_train --train_dataset rem-40 --do_eval --model_type albert --model_name_or_path albert/albert-base-v2

rm -r cache_dir
python finetune.py --do_train --train_dataset rem-20 --do_eval --model_type albert --model_name_or_path albert/albert-base-v2