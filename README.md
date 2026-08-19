\# AI Engineering Tutor (Local LLM Fine-Tuning)



An ongoing project to train and fine-tune a local Large Language Model (LLM) to act as a specialized teaching assistant for University-level Computer Engineering coursework using PyTorch and CUDA.



\## Phase 1: Data Pipeline

\- `extract\_pdfs.py`: Extracts and cleans text from syllabus and textbook PDFs using `PyMuPDF`.

\- `prepare\_dataset.py`: Uses a sliding-window algorithm to chunk text and format it into Alpaca-style JSONL pairs for PyTorch ingestion.



\### Setup

1\. Place PDFs in `./source\_pdfs`

2\. Run `python extract\_pdfs.py`

3\. Run `python prepare\_dataset.py`

