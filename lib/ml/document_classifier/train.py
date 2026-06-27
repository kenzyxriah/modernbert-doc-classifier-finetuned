import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from decouple import config
from utils import logger

BASE_MODEL = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
HF_REPO_ID = "KenzyXriah/modernbert-doc-classifier"
OUTPUT_DIR = "scratch/checkpoints"

def train_model(dataset: Dataset, id2label: dict, epochs: int = 3, batch_size: int = 8, max_length: int = 1024, output_dir: str = None, push_to_hub: bool = False)-> None:
    """
    Fine-tunes the base ModernBERT model on a provided dataset and saves the resulting weights to the configured directory.
    
    Args:
        dataset (Dataset): Hugging Face Dataset with 'text' and 'label' (int) columns.
        id2label (dict): Mapping from integer label IDs to string class names.
                         e.g., {0: "Public", 1: "Internal", 2: "Confidential", 3: "Restricted"}
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        max_length (int): Maximum sequence length for tokenization.
        output_dir (str, optional): Directory to save the fine-tuned model and checkpoints.
        push_to_hub (bool): Whether to instantly push the final trained model to the Hugging Face Hub.
    """
    target_output_dir = output_dir if output_dir else OUTPUT_DIR
    logger.info("Starting training pipeline...")
    
    num_labels = len(id2label)
    label2id = {v: k for k, v in id2label.items()}
    
    try:
        AutoConfig.from_pretrained(HF_REPO_ID, token=config("HF_TOKEN"))
        model_source = HF_REPO_ID
        logger.info(f"Found existing model on Hub! Resuming continuous training from {model_source}...")
    except Exception:
        model_source = BASE_MODEL
        logger.info(f"No existing model on Hub. Starting fresh training from {model_source}...")
    
    logger.info(f"Loading model and tokenizer from {model_source}...")
    tokenizer = AutoTokenizer.from_pretrained(model_source, token=config("HF_TOKEN"))
    
    model_config = AutoConfig.from_pretrained(model_source, token=config("HF_TOKEN"))
    model_config.num_labels = num_labels
    model_config.id2label = id2label
    model_config.label2id = label2id
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, 
        config=model_config,
        token=config("HF_TOKEN"),
        ignore_mismatched_sizes=True
    ).to(torch.float32)
    
    logger.info("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=target_output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        logging_steps=10,
        push_to_hub=push_to_hub,
        hub_model_id=HF_REPO_ID,
        hub_token=config("HF_TOKEN")
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator
    )
    
    resume = True if os.path.exists(target_output_dir) else False 
    try:
        trainer.train(resume_from_checkpoint=resume)
    except ValueError:
        trainer.train()
    
    if push_to_hub:
        logger.info(f"Pushing fine-tuned model to Hugging Face Hub ({HF_REPO_ID})...")
        trainer.push_to_hub()
        logger.info(f"Training complete! Model successfully pushed to {HF_REPO_ID}.")
    else:
        logger.info(f"Saving fine-tuned model to {target_output_dir}...")
        trainer.save_model(target_output_dir)
        logger.info(f"Training complete! Model saved to {target_output_dir}.")
