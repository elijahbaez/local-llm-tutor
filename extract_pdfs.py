import os
import pymupdf
import re

def clean_text(text):
    """Removes weird artifacts, extra spaces, and standardizes newlines."""
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Remove non-ascii or weird hidden PDF characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove excessive spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def extract_text_from_pdfs(pdf_dir):
    """Extracts and cleans text from all PDFs in a directory."""
    all_text = ""
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            filepath = os.path.join(pdf_dir, filename)
            print(f"Processing: {filename}")
            
            # Open PDF
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                raw_text = page.get_text("text")
                cleaned = clean_text(raw_text)
                
                # Simple heuristic to skip Table of Contents or blank pages
                if len(cleaned) > 100: 
                    all_text += cleaned + "\n\n"
                    
    return all_text

if __name__ == "__main__":
    pdf_directory = "./source_pdfs"
    if not os.path.exists(pdf_directory):
        print(f"Directory {pdf_directory} not found!")
    else:
        corpus = extract_text_from_pdfs(pdf_directory)
        with open("raw_corpus.txt", "w", encoding="utf-8") as f:
            f.write(corpus)
        print("Extraction complete. Saved to raw_corpus.txt")