# RAG Pipeline: OCR vs Non-OCR Comparison

A simplified RAG pipeline to test whether OCR improves retrieval abilities on documents.

## Prerequisites

### 1. Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

### 2. Install Poppler (for PDF to image conversion)

**macOS:**
```bash
brew install poppler
```

### 3. Get Groq API Key
1. Sign up at https://console.groq.com
2. Create an API key from the dashboard

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Set your Groq API key (optional - can also enter in the UI):
```bash
export GROQ_API_KEY='your-api-key-here'
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. In the web interface:
   - Enter your Groq API key (if not set as environment variable)
   - Upload a PDF or image document
   - Click "Process Document"
   - Ask questions in either tab to compare OCR vs non-OCR results

## How It Works

### Without OCR (Tab 1)
- Extracts text directly from PDFs using PyPDF2
- Works well for PDFs with embedded text
- Faster processing
- May fail on scanned documents or images

### With OCR (Tab 2)
- Converts PDF pages to images
- Uses Tesseract OCR to extract text
- Works on scanned documents and images
- Slower but more versatile

### RAG Pipeline
1. **Chunking**: Splits text into 500-word chunks
2. **Vectorization**: Uses TF-IDF for simple vector embeddings
3. **Retrieval**: Finds top 3 most relevant chunks using cosine similarity
4. **Generation**: Sends context + question to Groq's LLaMA model for answer

## Test Documents

For best comparison results, try:
- **Text-based PDFs**: Should work well in both tabs
- **Scanned PDFs**: Should only work well with OCR enabled
- **Images with text**: Only work with OCR enabled

## File Structure

```
.
├── app.py              # Streamlit frontend
├── rag_pipeline.py     # RAG backend logic
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Troubleshooting

**"Tesseract not found" error:**
- Verify installation: `which tesseract`
- Add to PATH if needed

**Groq API errors:**
- Check your API key is valid
- Verify you have credits available

**PDF processing fails:**
- Ensure poppler is installed: `which pdftoimage`

## Notes

- This is a simplified pipeline using TF-IDF for vectors (not true embeddings)
- For production use, consider using proper embedding models (OpenAI, Sentence Transformers)
- Chunk size and retrieval parameters can be adjusted in `rag_pipeline.py`