import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL = "Qwen/Qwen3.8-9B-Instruct"
ADAPTER_DIR = "./qwen-tutor-lora"
MERGED_DIR = "./qwen-tutor-merged"

def merge():
    if not os.path.exists(ADAPTER_DIR):
        print(f"Skipping merge: {ADAPTER_DIR} not found. (Run Phase 3 training first)")
        return

    print("Loading base model to CPU memory...")
    # We load to CPU to avoid VRAM spikes when adding weights together
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Merging weights (this requires significant RAM)...")
    # This bakes the LoRA connections permanently into the base model
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR}...")
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print("Merge complete!")

if __name__ == "__main__":
    # Uncomment to actually run the merge (requires successful training output)
    # merge()
    print(f"Simulation: Merged model would be saved to {MERGED_DIR}")