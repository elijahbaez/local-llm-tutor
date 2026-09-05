import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# Using the optimal 9B model we settled on in Phase 2
MODEL_ID = "Qwen/Qwen3.8-9B-Instruct"
OUTPUT_DIR = "./qwen-tutor-lora"

def format_instruction(example):
    """Formats our JSONL data into a single training string."""
    return f"Instruction: {example['instruction']}\n\nConcept: {example['output']}"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    # Load the JSONL file we made in Phase 1
    if not os.path.exists("training_data.jsonl"):
        raise FileNotFoundError("training_data.jsonl not found. Run Phase 1 scripts first.")
    
    dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

    print("Configuring 8-bit Quantization...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for training (freezes base weights, casts LayerNorms to fp32)
    model = prepare_model_for_kbit_training(model)

    print("Injecting LoRA Adapters...")
    # r=16 and alpha=32 is a standard starting point for domain adaptation (tutoring)
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Setting up Trainer...")
    # Optimizing batch sizes specifically for a 16GB RTX 5070 Ti
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,  # Keep this very small to fit in 16GB VRAM
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate a batch size of 8
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=200,                  # Short run for our prototype (change for full training)
        optim="paged_adamw_8bit",       # Highly memory efficient optimizer
        save_strategy="steps",
        save_steps=50,
        report_to="none"                # Bypassing wandb login for local testing
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=lambda example: [format_instruction(ex) for ex in example] if isinstance(example, list) else [format_instruction(example)],
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting Training! (This would take hours on a full dataset...)")
    # Uncomment the line below to ACTUALLY run training locally
    # trainer.train()
    
    print(f"Saving final adapter weights to {OUTPUT_DIR}...")
    # trainer.model.save_pretrained(OUTPUT_DIR)
    # tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()