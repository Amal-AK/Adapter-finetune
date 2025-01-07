import argparse
import logging
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.metrics import recall_score, precision_score, f1_score  # or remove if not needed for summarization

# If you have custom utilities or classes, import them here
# from utilities import EarlyStopper, set_seed, ...
# from optimization import regularized_evolution, ...
# from dataset import TextDatasetSummarization  # custom dataset for code summarization
# from OpenDelta.opendelta import AdapterModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


################################################################################
#                             Summarization Trainer
################################################################################
def train_summarization(args, model, tokenizer, train_dataloader, eval_dataloader, test_dataloader=None):
    """
    Train the T5 model for code summarization.
    """
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * 0.1),
        num_training_steps=max_steps
    )

    logger.info("***** Running training for code summarization *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", max_steps)

    model.zero_grad()
    # Example: early-stopping mechanism (if you want to adapt from your code)
    # early_stopper = EarlyStopper(patience=3, min_delta=0.01)
    
    best_val_loss = float("inf")
    results = {}

    for epoch in range(args.num_train_epochs):
        model.train()
        train_losses = []
        
        # Training loop
        for step, batch in enumerate(train_dataloader):
            # batch should contain input_ids, attention_mask, labels, etc.
            # Adjust to however your dataset returns samples
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            train_losses.append(loss.item())

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                logger.info(
                    "Epoch %d Step %d | Train Loss: %.4f",
                    epoch, step, np.mean(train_losses)
                )

        # End of epoch: gather stats
        avg_epoch_loss = round(float(np.mean(train_losses)), 4)
        logger.info("Epoch %d completed. Average Train Loss = %.4f", epoch, avg_epoch_loss)

        # Evaluate on validation set
        eval_result = evaluate_summarization(args, model, tokenizer, eval_dataloader)
        val_loss = eval_result["eval_loss"]
        logger.info("  Validation Loss = %.4f", val_loss)

        # Save stats
        results.setdefault("train_loss", []).append(avg_epoch_loss)
        results.setdefault("eval_loss", []).append(val_loss)

        # If validation improved => save "best" model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info("\n%s", "*" * 40)
            logger.info("  Best validation loss: %.4f", best_val_loss)
            logger.info("%s\n", "*" * 40)

            if not args.do_optimization:
                # test the model
                if test_dataloader:
                    test_result = test_summarization(args, model, tokenizer, test_dataloader)
                    logger.info("Test results (best model so far): %s", test_result)

                # Save best model
                save_best_model(model, args, "models/best_model_summarization")

        # Example early stopping
        # if early_stopper.early_stop(val_loss):
        #     logger.info("Early stopping triggered...")
        #     break

    # After all epochs, optionally test final model
    if not args.do_optimization and test_dataloader:
        test_result = test_summarization(args, model, tokenizer, test_dataloader)
        logger.info("Final Test results: %s", test_result)

        save_best_model(model, args, "models/final_model_summarization")

    return results


################################################################################
#                            Summarization Evaluator
################################################################################
def evaluate_summarization(args, model, tokenizer, eval_dataloader):
    """
    Evaluate the T5 model on a validation set.
    Returns a dict with "eval_loss" (and optional metrics).
    """
    logger.info("\n***** Running evaluation for code summarization *****")
    logger.info("  Num examples = %d", len(eval_dataloader.dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, nb_eval_steps = 0.0, 0

    # Optionally store generated predictions if you want ROUGE, BLEU, etc.
    # from datasets import load_metric
    # rouge = load_metric("rouge")

    for batch in eval_dataloader:
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            eval_loss += loss.item()
            nb_eval_steps += 1

            # If you want to actually generate predictions to compute ROUGE:
            # generated_ids = model.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     max_length=50
            # )
            # dec_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # rouge.add_batch(predictions=dec_preds, references=dec_labels)

    eval_loss = eval_loss / nb_eval_steps

    # Example ROUGE computation (if you used the code above):
    # final_rouge = rouge.compute()
    # print("ROUGE metrics:", final_rouge)

    result = {
        "eval_loss": round(eval_loss, 4)
        # "rouge1": final_rouge["rouge1"].mid.fmeasure if final_rouge else None,
        # "rouge2": final_rouge["rouge2"].mid.fmeasure if final_rouge else None,
        # "rougeL": final_rouge["rougeL"].mid.fmeasure if final_rouge else None
    }
    return result


################################################################################
#                            Summarization Tester
################################################################################
def test_summarization(args, model, tokenizer, test_dataloader):
    """
    Generate summaries for the test set and compute metrics.
    """
    logger.info("***** Running testing for code summarization *****")
    logger.info("  Num examples = %d", len(test_dataloader.dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    predictions = []
    references = []

    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)

        with torch.no_grad():
            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_target_length,   # or whatever
                num_beams=4
            )

        # Decode predictions
        pred_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        ref_summaries = tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions.extend(pred_summaries)
        references.extend(ref_summaries)

    # Compute final metrics (e.g., ROUGE, BLEU, etc.)
    # from datasets import load_metric
    # rouge = load_metric("rouge")
    # for pred, ref in zip(predictions, references):
    #     rouge.add(prediction=pred, reference=ref)
    # final_rouge = rouge.compute()

    # For demonstration, we’ll just return references/predictions count
    result = {
        "num_predictions": len(predictions),
        # "rouge1": final_rouge["rouge1"].mid.fmeasure,
        # "rouge2": final_rouge["rouge2"].mid.fmeasure,
        # "rougeL": final_rouge["rougeL"].mid.fmeasure
    }
    logger.info("Test set generation done. Example prediction:")
    logger.info("  Prediction: %s", predictions[0] if predictions else None)
    logger.info("  Reference:  %s", references[0] if references else None)
    logger.info("Test results: %s", result)
    return result


################################################################################
#                       Save & Utility Functions
################################################################################
def save_best_model(model, args, checkpoint_prefix="best_model"):
    """
    Save the current model to disk.
    """
    output_dir = os.path.join(args.output_dir, checkpoint_prefix)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)


def set_seed(seed=42):
    """
    Set all random seeds for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


################################################################################
#                                  Main
################################################################################
def main():
    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument("--train_data_file", default="./datasets/summarization/train.jsonl", type=str,
                        help="Path to the train dataset.")
    parser.add_argument("--eval_data_file", default="./datasets/summarization/valid.jsonl", type=str,
                        help="Path to the validation dataset.")
    parser.add_argument("--test_data_file", default="./datasets/summarization/test.jsonl", type=str,
                        help="Path to the test dataset.")

    parser.add_argument("--output_dir", default="./", type=str,
                        help="Where to save models/checkpoints.")

    # Model hyperparams
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5p-220m", type=str,
                        help="Pretrained CodeT5p model checkpoint.")
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="Max sequence length for the code input.")
    parser.add_argument("--max_target_length", default=64, type=int,
                        help="Max sequence length for the summary output.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="Learning rate.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Training batch size.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Eval batch size.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total epochs to train.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Gradient clipping.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    # Flow control
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to run testing on the test set.")
    parser.add_argument("--do_optimization", action="store_true",
                        help="Set True if you are doing adapter optimization, else normal training.")

    args = parser.parse_args()
    set_seed(args.seed)

    # GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    logger.info("Using device %s", device)

    # Load config, tokenizer, model
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True)

    model.to(device)

    # --- If you want to apply adapters (OpenDelta) ---
    # delta_model = AdapterModel(backbone_model=model, ...)
    # delta_model.freeze_module(exclude=["adapter"], set_state_dict=True)
    # Or apply whatever logic you have for randomly generating adapter configs, etc.

    # Create Datasets & Dataloaders
    # You need a custom TextDatasetSummarization or similar
    train_dataloader = None
    eval_dataloader = None
    test_dataloader = None

    if args.do_train:
        # Example: create DataLoader for training
        train_dataset = TextDatasetSummarization(
            tokenizer, args.train_data_file, max_source_length=args.max_source_length, max_target_length=args.max_target_length
        )
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.do_eval:
        eval_dataset = TextDatasetSummarization(
            tokenizer, args.eval_data_file, max_source_length=args.max_source_length, max_target_length=args.max_target_length
        )
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_test:
        test_dataset = TextDatasetSummarization(
            tokenizer, args.test_data_file, max_source_length=args.max_source_length, max_target_length=args.max_target_length
        )
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    # If you're doing adapter search (like your defect script), you'd call something like:
    # if args.do_optimization:
    #     history, population, best_of_all, stats = regularized_evolution(args, config, train_dataloader, eval_dataloader)

    # Otherwise, do normal training
    if args.do_train:
        train_summarization(args, model, tokenizer, train_dataloader, eval_dataloader, test_dataloader)

    # Evaluate if needed
    if args.do_eval and not args.do_train:
        eval_result = evaluate_summarization(args, model, tokenizer, eval_dataloader)
        logger.info("Eval results: %s", eval_result)

    # Test if needed
    if args.do_test and not args.do_train:
        test_result = test_summarization(args, model, tokenizer, test_dataloader)
        logger.info("Test results: %s", test_result)


###############################################################################
# Example custom dataset for Summarization
###############################################################################
class TextDatasetSummarization(torch.utils.data.Dataset):
    """
    A simple dataset that reads a JSONL file with each line containing:
    {
      "code": "...",
      "summary": "..."
    }
    and tokenizes them for T5. Adjust as needed.
    """
    def __init__(self, tokenizer, file_path, max_source_length=256, max_target_length=64):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # Load data
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = line.strip()
                if not data:
                    continue
                # Parse line (assuming JSON) ...
                import json
                example = json.loads(data)
                code = example["code"]
                summary = example["summary"]
                
                self.examples.append((code, summary))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        code, summary = self.examples[idx]

        # Tokenize input code
        source_encoding = self.tokenizer(
            code,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target summary
        target_encoding = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Return a dictionary for collate_fn
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }


if __name__ == "__main__":
    main()
