import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3.8-9B-Instruct"

def load_model():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print(f"Loading model {MODEL_ID} to CUDA with 8-bit Quantization...")
    # FIX: Implement 8-bit quantization to prevent CUDA Out of Memory (OOM)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config
    )
    return tokenizer, model

def generate_response(prompt, tokenizer, model):
    messages = [
        {"role": "system", "content": "You are a helpful Computer Engineering AI Tutor."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    print("Generating response...")
    outputs = model.generate(**inputs, max_new_tokens=200)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's reply from the chat template
    return response.split("<|im_start|>assistant\n")[-1]

if __name__ == "__main__":
    tokenizer, model = load_model()
    
    test_prompt = "Explain the difference between a RISC and CISC architecture."
    print(f"\nPrompt: {test_prompt}\n")
    print("-" * 40)
    print(generate_response(test_prompt, tokenizer, model))