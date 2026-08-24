import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Using the optimal 9B model for a 16GB RTX 5070 Ti
MODEL_ID = "Qwen/Qwen3.8-9B-Instruct"

def load_model():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print(f"Loading model {MODEL_ID} to CUDA...")
    # Standard loading - this requires ~18GB+ VRAM for 9B parameters and might crash
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return tokenizer, model

def generate_response(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("Generating response...")
    outputs = model.generate(**inputs, max_new_tokens=150)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    tokenizer, model = load_model()
    
    test_prompt = "Explain the difference between a RISC and CISC architecture."
    print(f"\nPrompt: {test_prompt}\n")
    print("-" * 40)
    print(generate_response(test_prompt, tokenizer, model))