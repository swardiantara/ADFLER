# Train on DroneOvs 100%, evaluate on Original Test (DT1)
rm -r cache_dir
python finetune.py --do_train --train_dataset rem-100 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario rq3-verification --seed 42
rm -r cache_dir
python finetune.py --do_train --train_dataset rem-100 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario rq3-verification --seed 123
rm -r cache_dir
python finetune.py --do_train --train_dataset rem-100 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario rq3-verification --seed 99
rm -r cache_dir
python finetune.py --do_train --train_dataset rem-100 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario rq3-verification --seed 17
rm -r cache_dir
python finetune.py --do_train --train_dataset rem-100 --do_eval --model_type bert --model_name_or_path bert-base-cased --scenario rq3-verification --seed 67