rm -r cache_dir
python finetune.py --do_eval --eval_dataset ori1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset ori2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset ori3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset ori4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset ori5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset ori5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-80 --eval_dataset ori1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset ori2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset ori3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset ori4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset ori5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-60 --eval_dataset ori1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset ori2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset ori3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset ori4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset ori5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-40 --eval_dataset ori1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset ori2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset ori3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset ori4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset ori5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-20 --eval_dataset ori1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset ori2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset ori3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset ori4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset ori5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --eval_dataset rem1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset rem2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset rem3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset rem4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset rem5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset rem5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-80 --eval_dataset rem1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset rem2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset rem3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset rem4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset rem5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-60 --eval_dataset rem1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset rem2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset rem3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset rem4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset rem5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-40 --eval_dataset rem1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset rem2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset rem3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset rem4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset rem5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-20 --eval_dataset rem1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset rem2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset rem3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset rem4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset rem5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --eval_dataset low1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset low2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset low3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset low4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --eval_dataset low5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-100 --eval_dataset low1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset low2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset low3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset low4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-100 --eval_dataset low5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-80 --eval_dataset low1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset low2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset low3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset low4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-80 --eval_dataset low5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-60 --eval_dataset low1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset low2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset low3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset low4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-60 --eval_dataset low5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-40 --eval_dataset low1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset low2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset low3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset low4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-40 --eval_dataset low5 --model_type distilbert --model_name_or_path distilbert-base-cased

python finetune.py --do_eval --train_dataset rem-20 --eval_dataset low1 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset low2 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset low3 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset low4 --model_type distilbert --model_name_or_path distilbert-base-cased
python finetune.py --do_eval --train_dataset rem-20 --eval_dataset low5 --model_type distilbert --model_name_or_path distilbert-base-cased