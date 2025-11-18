# Smart RAG Document Analyzer

## 1. High-Level Overview

**Smart RAG Document Analyzer** is an advanced tool designed to intelligently understand and answer questions about complex documents. Unlike simple text search, this application can comprehend content within tables, charts, and images, providing users with accurate, context-aware answers.

It addresses the critical need of extracting insights from data-rich reports, presentations, and technical documents that mix text with complex visuals. By leveraging a sophisticated AI pipeline, it saves significant time and effort in document analysis.

## 2. Key Features

*   **Advanced Chart & Image Analysis**: Utilizes local, privacy-preserving Vision Models (e.g., [Moondream2](https://huggingface.co/vikhyatk/moondream2), [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)) to interpret charts, graphs, and images within documents. It extracts trends, data points, and key insights from visuals.
*   **Comprehensive Document Support**: Ingests and understands various enterprise formats, including PDF (`.pdf`), Microsoft Word (`.docx`), and PowerPoint (`.pptx`).
*   **Intelligent Content Chunking**: Employs a "smart chunking" strategy that preserves the semantic integrity of the content. It ensures that related pieces of information, especially text and its corresponding chart descriptions, are kept together for better context.
*   **Dual-Method Chart Detection**: Maximizes accuracy by using a two-pronged approach to find visuals:
    *   **[Microsoft TATR (Table Transformer)](https://huggingface.co/docs/transformers/main/en/model_doc/table-transformer)**: An AI model that excels at identifying the structure of tables and charts.
    *   **Heuristic Detection**: A rules-based algorithm that analyzes pixel data to find charts that machine learning models might miss.
*   **High-Speed, Accurate Search**: Leverages **[FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)**, a powerful vector database, to perform lightning-fast searches and find the most relevant information to answer user queries.
*   **Interactive User Interface**: A user-friendly web interface built with Streamlit allows users to easily upload documents, manage sessions, and interact with the AI in a chat-based format.

## 3. How It Works: The RAG Pipeline

The application follows a systematic process to convert a raw document into a searchable knowledge base.

1.  **Ingestion & Parsing**: The user uploads a document. The system identifies the file type and begins parsing.
2.  **Visual Detection & Analysis**: It scans each page for charts and images using both Microsoft TATR and heuristic methods. Detected visuals are sent to a local Vision AI model, which generates detailed text descriptions.
3.  **Markdown Conversion**: The entire document—including the original text and the new AI-generated chart descriptions—is converted into a unified Markdown format. This preserves the document's structure and makes it easy to process.
4.  **Smart Chunking**: The Markdown content is broken down into small, semantically related "chunks." This method ensures that important context is not lost by splitting a paragraph or separating a chart's description from its data.
5.  **Embedding & Indexing**: Each chunk is converted into a numerical representation (an "embedding") using an advanced sentence-transformer model. These embeddings are stored in a **FAISS** index, which is highly optimized for fast similarity searches.
6.  **Query & Synthesis**: When a user asks a question, the system searches the FAISS index to find the most relevant chunks of text. This context is then provided to a powerful Large Language Model (LLM) via the Groq API, which synthesizes a concise and accurate answer based *only* on the information from the document.

## 4. Core Technologies

*   **Document Processing**: PyMuPDF (fitz), `python-docx`, `python-pptx`
*   **Chart/Image Detection**: Microsoft TATR (Table Transformer), Heuristic Algorithms
*   **Vision Models**: [Moondream2](https://huggingface.co/vikhyatk/moondream2), Qwen3-VL-2B, InternVL3.5-1B (run locally)
*   **Text Embedding**: Sentence-Transformers
*   **Vector Storage & Search**: FAISS (Facebook AI Similarity Search)
*   **Chunking Strategy**: Custom "Smart Chunking" logic
*   **LLM for Q&A**: Groq API (Llama 4)
*   **Application Framework**: Streamlit

## 5. Project Structure

The project is organized into a modular structure to separate concerns, making it maintainable and scalable.

```
/
├── data/                  # Stores persistent data like indexes, chunks, and logs
│   ├── chunks/
│   ├── faiss_indexes/
│   └── file_steps/
├── potential_charts/      # Output directory for extracted chart/image files
├── src/                   # Main source code
│   ├── app/               # Streamlit UI components (sidebar, main content)
│   ├── core/              # Core RAG pipeline logic (parsing, chunking, indexing)
│   ├── services/          # Clients for external APIs (e.g., Groq)
│   ├── utils/             # Helper utilities (chart detection, database management)
│   └── vision/            # Vision model integrations
├── static/                # Static assets like CSS
├── main.py                # Entry point to run the Streamlit application
└── README.md              # This file
```

## 6. Getting Started

1.  **Set Up Environment**: Ensure all dependencies from `requirements.txt` are installed.
2.  **API Keys**: Set the `GROQ_API_KEY` environment variable.
3.  **Run Application**: Execute the following command in your terminal:
    ```bash
    streamlit run main.py
    ```
4.  **Interact**: Open the provided URL in your browser to start analyzing documents.