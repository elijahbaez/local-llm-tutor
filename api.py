from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

app = FastAPI(title="AI Tutor API", description="Local API for Fine-Tuned Qwen Tutor")

# In production, this would point to "./qwen-tutor-merged"
# For local testing without a full training run, we point it to the base model
MODEL_PATH = "Qwen/Qwen3.8-9B-Instruct" 

tokenizer = None
model = None

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

@app.on_event("startup")
def startup_event():
    global tokenizer, model
    print("Initializing AI Tutor on CUDA...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto"
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet.")

    messages = [
        {"role": "system", "content": "You are a highly intelligent Computer Engineering AI Tutor."},
        {"role": "user", "content": request.prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=request.max_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the output to only return the AI's reply
    reply = response.split("<|im_start|>assistant\n")[-1].strip()
    return {"response": reply}

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Run API server using: uvicorn api:app --reload")