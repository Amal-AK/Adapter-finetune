python3 main_clone.py \
    --train_data_file=./datasets/dataset_clone/train.txt \
    --output_dir=./ \
    --eval_data_file=./datasets/dataset_clone/valid.txt \
    --test_data_file=./datasets/dataset_clone/test.txt \
    --model_name_or_path=Salesforce/codet5-base \
    --tokenizer_name=Salesforce/codet5-base \
    --num_classes 1 \
    --nl_length 128 \
    --code_length 512 \
    --do_optimization True \
    --do_test False \
    --do_train False \
    --do_eval False \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --train_data_rate_clone 1.0 \
    --learning_rate 1e-4 \
    --population_size 50 \
    --sample_size 10 \
    --nb_samples 10000 \
    --cycles 30 \
    --max_grad_norm 1.0 \
    --num_train_epochs 5 \
    --optimization_history_file=logs_optim/codet5_optim_history_clone.txt \
    --stats_file=logs_optim/codet5_stats_clone.json \
    --seed 42 2>&1 | tee ./logs_optim/codet5_clone_detection.log
