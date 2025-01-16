python3 main_clone.py \
    --train_data_file=./datasets/dataset_clone/train.txt \
    --output_dir=./ \
    --eval_data_file=./datasets/dataset_clone/valid.txt \
    --test_data_file=./datasets/dataset_clone/test.txt \
    --model_name_or_path=Salesforce/codet5p-220m \
    --tokenizer_name=Salesforce/codet5p-220m \
    --num_classes 1 \
    --nl_length 128 \
    --code_length 512 \
    --do_train True \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --train_data_rate_clone 0.1 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --num_train_epochs 15 \
    --seed 42 2>&1 | tee ./baselines/codet5p_clone_adapter_default.log
