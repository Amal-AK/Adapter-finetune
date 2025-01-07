from main_defect import train_defect , evaluate_defect , test_defect
from myOpenDelta.opendelta import LoraModel , AdapterModel , PrefixModel
from utilities import *
import torch.nn as nn
import argparse
import logging
import os
import sys
import torch
import numpy as np
from model import Model_classification
from transformers import ( RobertaConfig, RobertaTokenizer ,RobertaModel ,  T5ForConditionalGeneration)
from sklearn.metrics import recall_score, precision_score, f1_score
from utilities import *
from optimization import *
from transformers import T5Config 

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("my_logger")
#logging.getLogger("transformers.modeling_utils").setLevel(logging.INFO)





def main():



    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file_defect", default="./datasets/dataset_defect/train.jsonl", type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--eval_data_file_defect", default="./datasets/dataset_defect/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file_defect", default="./datasets/dataset_defect/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    
    parser.add_argument("--PEFT_method", default='adapter', type=str,
                        help="which peft method to use ")
    
    parser.add_argument("--full_finetune", default=False, type=bool,
                        help="if the model should be finetuned in full option or peft methods ")
    
    parser.add_argument("--output_dir", default='./', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_classes", default=1, type=int,
                        help="The number of classes for the classification model")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    parser.add_argument("--model_name_or_path", default='microsoft/codebert-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--do_optimization", action='store_true',
                        help="Whether to run adapter optimization")  
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.") 
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--train_data_rate", default=1.0, type= float,
                        help="Data size for train")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--local_rank', default=-1 ,type=int,
                        help="random seed for initialization")
    parser.add_argument('--population_size', default=20 ,type=int,
                        help="population size on the evolutionary optimization algorithm")
    
    parser.add_argument('--sample_size', default=10 ,type=int,
                        help="sample size on the evolutionary optimization algorithm")
    
    parser.add_argument('--cycles', default=20 ,type=int,
                        help="number of cycles on the evolutionary optimization algorithm")
    
    parser.add_argument('--optimization_history_file', default=None ,type=str,
                        help="saving the history of optimization")
    
    
    args = parser.parse_args()
    set_seed(seed=args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = [ 1 if torch.cuda.is_available() else 0][0]
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    config = RobertaConfig.from_pretrained(args.model_name_or_path , num_labels = args.num_classes)
    

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path,config=config)  

    if not args.full_finetune : 

        if args.PEFT_method != None : 

            #delta_model = AdapterModel(backbone_model=model,modified_modules=['attention','attention.self', 'intermediate', '[r](\d)+\.output'],bottleneck_dim=[32] )  
      

            if args.PEFT_method == "adapter" : 
                delta_model = AdapterModel(backbone_model=model,bottleneck_dim=[32] )  
            elif args.PEFT_method == "lora" : 
                delta_model = LoraModel(backbone_model=model )  
            elif args.PEFT_method == "prefix" : 
                delta_model = PrefixModel(backbone_model=model ) 

            delta_model.freeze_module(exclude=["deltas" ])
            delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)

        else :
            print("Please specify which method to use for the finetuning")
            sys.exit(1)
             
    
        
    model = Model_classification( model , config)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.to(args.device)

     
    if args.do_train:
            results = train_defect(args , model ,tokenizer)
            print("train results", results)


       
       
if __name__ == "__main__":
    main()