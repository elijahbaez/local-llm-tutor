import json
import os
from tqdm import tqdm

def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into sliding windows (chunks) of words."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 100:  # Only keep substantial chunks
            chunks.append(chunk)
            
    return chunks

def create_jsonl_dataset(chunks, output_filename="training_data.jsonl"):
    """Formats chunks into Alpaca instruction-response pairs."""
    with open(output_filename, 'w', encoding='utf-8') as f:
        for chunk in tqdm(chunks, desc="Creating JSONL Dataset"):
            # For our initial pre-training/fine-tuning, we tell the model to act as a tutor
            data_point = {
                "instruction": "Explain the following computer engineering concept in detail.",
                "input": "",
                "output": chunk
            }
            f.write(json.dumps(data_point) + "\n")

if __name__ == "__main__":
    if not os.path.exists("raw_corpus.txt"):
        print("raw_corpus.txt not found. Run extract_pdfs.py first.")
    else:
        with open("raw_corpus.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        print("Chunking text...")
        text_chunks = chunk_text(raw_text)
        print(f"Generated {len(text_chunks)} chunks.")
        
        create_jsonl_dataset(text_chunks)
        print("Dataset ready: training_data.jsonl")