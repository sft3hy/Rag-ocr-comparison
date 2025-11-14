import streamlit as st
from rag_enhanced import SmartRAG
import os
import time
import tempfile
from pathlib import Path
from PIL import Image


from custom_css import custom

# Page config with custom theme
st.set_page_config(
    page_title="Smart RAG Document Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, professional look with animations
st.markdown(
    custom,
    unsafe_allow_html=True,
)

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None
if "chart_dir" not in st.session_state:
    st.session_state.chart_dir = None

# with st.sidebar:
#     detector_model = st.radio(
#         "Choose a Chart/Image detection model",
#         ["yolo", "tatr"],
#         captions=[
#             "foduucom/table-detection-and-extraction",
#             "microsoft/table-transformer-detection",
#         ],
#     )

detector_model = "heuristic"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_chart_output_dir(filename: str, model: str) -> Path:
    """Create a unique output directory path for charts."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Use Path for cleaner path handling, stem removes extension
    safe_name = Path(filename).stem.replace(" ", "_")
    dir_name = f"{timestamp}_{model}_{safe_name}"
    return Path("potential_charts") / dir_name


def find_actual_chart_directory() -> Path | None:
    """Search for the most recently created chart directory in case SmartRAG created its own."""
    potential_charts = Path("potential_charts")
    if not potential_charts.exists():
        return None

    # Get all subdirectories sorted by modification time (most recent first)
    subdirs = [d for d in potential_charts.iterdir() if d.is_dir()]
    if not subdirs:
        return None

    # Return the most recently modified directory
    return max(subdirs, key=lambda d: d.stat().st_mtime)


def get_all_chart_images(chart_dir: Path | None) -> list[Path]:
    """Get all chart PNG files from the directory, sorted by page number."""
    if chart_dir is None or not chart_dir.exists():
        return []

    # Use pathlib's glob - much cleaner than os.listdir
    chart_files = list(chart_dir.glob("*.png"))

    # Sort by page number extracted from filename
    return sorted(chart_files, key=extract_page_number)


def extract_page_number(filepath: Path) -> int:
    """Extract page number from filename like 'page5_chart1.png'."""
    import re

    match = re.search(r"page(\d+)", filepath.name)
    return int(match.group(1)) if match else float("inf")


def get_charts_for_page(chart_dir: Path, page: int) -> list[Path]:
    """Get all chart images for a specific page."""
    if not chart_dir.exists():
        return []

    # Use glob patterns - more reliable than manual string matching
    patterns = [f"page{page}_chart*.png", f"page{page}_embedded*.png"]

    charts = []
    for pattern in patterns:
        charts.extend(chart_dir.glob(pattern))

    # Remove duplicates while preserving order
    return list(dict.fromkeys(charts))


# ============================================================================
# HEADER
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        "<h1 style='text-align: center;'>🧠 Smart RAG Document Analyzer</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #e0e0ff; font-size: 1.2rem; margin-top: -1rem;'>AI-powered document analysis with OCR and vision understanding</p>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)

    groq_api_key = os.environ.get("GROQ_API_KEY", "")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file and groq_api_key:
        if st.button("🚀 Process Document", width="stretch"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Create output directory for charts FIRST
                chart_dir = get_chart_output_dir(uploaded_file.name, detector_model)
                chart_dir.mkdir(parents=True, exist_ok=True)

                # Store as absolute path to ensure it's accessible
                st.session_state.chart_dir = chart_dir.resolve()

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    st.session_state.temp_file_path = tmp_file.name

                progress_bar.progress(25)
                status_text.markdown("🔄 **Initializing Smart RAG system...**")
                time.sleep(0.3)
                status_text.markdown("📊 **Starting document analysis...**")
                time.sleep(0.3)

                # Initialize RAG system with absolute path
                st.session_state.rag_pipeline = SmartRAG(
                    output_dir=str(st.session_state.chart_dir),
                    chart_detector_model=detector_model,
                )

                # Progress callback
                def update_progress(current_page, total_pages):
                    progress = 25 + int((current_page / total_pages) * 65)
                    progress_bar.progress(progress)
                    status_text.markdown(
                        f"📊 **Processing page {current_page}/{total_pages}...**"
                    )
                    time.sleep(0.1)

                # Process document
                st.session_state.rag_pipeline.index_document(
                    st.session_state.temp_file_path,
                    chunk_size=500,
                    overlap=100,
                    progress_callback=update_progress,
                )

                # IMPORTANT: After processing, check if SmartRAG created a subdirectory
                # If your rag_enhanced.py still creates subdirectories, we need to find them
                pdf_basename = Path(uploaded_file.name).stem
                potential_subdir = st.session_state.chart_dir / pdf_basename

                if potential_subdir.exists() and list(potential_subdir.glob("*.png")):
                    # SmartRAG created a subdirectory - use it
                    st.session_state.chart_dir = potential_subdir
                    print(f"Charts found in subdirectory: {st.session_state.chart_dir}")
                else:
                    # Charts are in the main directory as expected
                    print(f"Charts in main directory: {st.session_state.chart_dir}")

                progress_bar.progress(90)
                status_text.markdown("🔍 **Building search index...**")
                time.sleep(0.3)

                progress_bar.progress(100)
                status_text.markdown("✅ **Processing complete!**")
                st.session_state.processing_complete = True

                # Debug: Print the directory we're checking
                print(f"Looking for charts in: {st.session_state.chart_dir}")
                print(f"Directory exists: {st.session_state.chart_dir.exists()}")

                if st.session_state.chart_dir.exists():
                    all_files = list(st.session_state.chart_dir.glob("*"))
                    # print(f"All files in directory: {[f.name for f in all_files]}")

                # Verify charts are in the expected location
                actual_charts = get_all_chart_images(st.session_state.chart_dir)
                print(f"Found {len(actual_charts)} chart images")

                time.sleep(0.5)

                # Show results
                if actual_charts:
                    status_text.markdown(f"📊 Found {len(actual_charts)} charts")
                else:
                    # If no charts found, check parent directory
                    parent_charts = list(
                        st.session_state.chart_dir.parent.glob("**/*.png")
                    )
                    if parent_charts:
                        status_text.markdown(
                            f"⚠️ No charts in expected location. Found {len(parent_charts)} charts in parent directories. Check console for paths."
                        )
                        print(
                            f"Charts found elsewhere: {[str(p) for p in parent_charts[:5]]}"
                        )
                    else:
                        status_text.markdown(f"⚠️ No charts detected in document")
                time.sleep(2)

                progress_bar.empty()
                status_text.empty()

                st.markdown(
                    "<div class='success-message'>✨ Document processed successfully!</div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error processing document: {str(e)}")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    # Technology explanations
    st.markdown("### 🔧 Technologies Used")

    with st.expander("📷 Tesseract OCR"):
        st.markdown(
            """
        **Optical Character Recognition Engine**
        
        Tesseract is an open-source OCR engine that extracts text from images and scanned documents. 
        
        **In this app:**
        - Extracts text from charts and images embedded in PDFs
        - Handles low-quality scans and non-native text
        - Provides confidence scores for extracted text
        """
        )

    with st.expander("👁️ Moondream2 Vision Model"):
        st.markdown(
            """
        **AI Vision Model for Chart Understanding**
        
        Moondream2 is a compact vision-language model that can understand and describe visual content.
        
        **In this app:**
        - Analyzes charts, graphs, and diagrams
        - Extracts trends, data points, and insights
        - Generates detailed descriptions of visual elements
        - Runs locally on your device (MPS/CUDA/CPU)
        """
        )
    with st.expander("📄 Table Transformer (TATR)"):
        st.markdown(
            """
        **AI Model for Table and Structure Detection**

        TATR (Table Transformer) is a sophisticated model from Microsoft, built on the DETR (DEtection TRansformer) architecture. It excels at identifying the structure and location of tables within documents.

        **In this app:**
        - Detects the precise location of tables and charts on a document page.
        - Uses an object detection approach to draw bounding boxes around potential visual data.
        - Is particularly effective at finding structured elements, which makes it a strong candidate for chart detection.
        """
        )
    with st.expander("🎯 YOLOv8 Object Detection"):
        st.markdown(
            """
        **Real-Time Object Detection Model**

        YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system known for its incredible speed and accuracy. YOLOv8 is the latest iteration in this popular model family.

        **In this app:**
        - Scans the document page to instantly identify regions that look like charts or tables.
        - Acts as a pluggable and highly efficient detector to locate visual data.
        - Can be fine-tuned on custom datasets to recognize specific types of charts, offering a path for even greater accuracy.
        """
        )

    with st.expander("🤖 Groq + Llama 3.3"):
        st.markdown(
            """
        **Fast Language Model Inference**
        
        Groq provides ultra-fast inference for large language models like Meta's Llama 3.3 70B.
        
        **In this app:**
        - Generates accurate answers to your questions
        - Synthesizes information from multiple sources
        - Provides context-aware responses
        - Runs at 750+ tokens/second
        """
        )

    with st.expander("🔍 Sentence Transformers"):
        st.markdown(
            """
        **Semantic Search & Embeddings**
        
        Creates dense vector representations of text for semantic similarity search.
        
        **In this app:**
        - Converts text chunks into vector embeddings
        - Enables semantic search (not just keyword matching)
        - Finds relevant context for your questions
        - Model: all-MiniLM-L6-v2
        """
        )

    with st.expander("⚡ FAISS Vector Database"):
        st.markdown(
            """
        **Efficient Similarity Search**
        
        Facebook AI Similarity Search - optimized library for fast vector search.
        
        **In this app:**
        - Stores document embeddings efficiently
        - Performs fast nearest-neighbor search
        - Retrieves most relevant chunks in milliseconds
        - Scales to millions of vectors
        """
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Supported Formats")
    st.markdown(
        """<div class='sidebarText'>
    - 📕 PDF Documents<br>
    - 🖼️ Scanned PDFs<br>
    - 📊 Documents with Charts/Graphs<br>
    - 📄 Text-heavy PDFs<br>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #fff; font-size: 0.8rem;'>Built with Streamlit • Powered by AI</p>",
        unsafe_allow_html=True,
    )

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

if not st.session_state.processing_complete:
    # Welcome screen
    st.markdown(
        """
    <div style='text-align: center; padding: 4rem 2rem; background: rgba(255, 255, 255, 0.05); border-radius: 20px; backdrop-filter: blur(10px);'>
        <h2 style='font-size: 2rem; margin-bottom: 1rem;'>👋 Welcome to Smart RAG</h2>
        <p style='font-size: 1.2rem; color: #e0e0ff; margin-bottom: 2rem;'>
            Upload a PDF to unlock advanced document analysis with AI
        </p>
        <div style='display: flex; justify-content: center; gap: 3rem; margin-top: 3rem; flex-wrap: wrap;'>
            <div class='metric-card' style='min-width: 200px;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>📊</div>
                <div style='font-weight: 600; font-size: 1.1rem;'>Chart Understanding</div>
                <div style='font-size: 0.9rem; color: #b0b0ff; margin-top: 0.5rem;'>AI vision analyzes graphs</div>
            </div>
            <div class='metric-card' style='min-width: 200px;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🔍</div>
                <div style='font-weight: 600; font-size: 1.1rem;'>OCR Extraction</div>
                <div style='font-size: 0.9rem; color: #b0b0ff; margin-top: 0.5rem;'>Extract text from images</div>
            </div>
            <div class='metric-card' style='min-width: 200px;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🎯</div>
                <div style='font-weight: 600; font-size: 1.1rem;'>Smart Search</div>
                <div style='font-size: 0.9rem; color: #b0b0ff; margin-top: 0.5rem;'>Semantic retrieval</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
else:
    # ============================================================================
    # QUESTION INTERFACE
    # ============================================================================

    question = st.text_input(
        "question",
        label_visibility="hidden",
        placeholder="💬 Ask a question about your document:",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ask_button = st.button("🎯 Get Answer", width="stretch")

    if ask_button and question:
        with st.spinner("🤔 Analyzing document and generating answer..."):
            try:
                query_result = st.session_state.rag_pipeline.query(question, top_k=5)

                if "error" in query_result:
                    st.error(f"❌ Error: {query_result['error']}")
                else:
                    answer = query_result["response"]
                    results = query_result["results"]

                    # Collect all chart images referenced in results
                    chart_images = []
                    for result in results:
                        if "[CHART DESCRIPTION" in result["text"]:
                            page = result["page"]
                            # Use utility function to get charts for this page
                            page_charts = get_charts_for_page(
                                st.session_state.chart_dir, page
                            )
                            chart_images.extend(page_charts)

                    # Remove duplicates while preserving order
                    chart_images = list(dict.fromkeys(chart_images))

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                    <div class='answer-container'>
                        <h3 style='margin-top: 0;'>💡 Answer</h3>
                        <p style='font-size: 1.1rem; line-height: 1.6; color: #ffffff;'>{answer}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Display images used in response
                    if chart_images:
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander(
                            f"🖼️ Images Used in Response ({len(chart_images)})"
                        ):
                            for img_path in chart_images:
                                # Get description from RAG pipeline
                                description = st.session_state.rag_pipeline.chart_descriptions.get(
                                    img_path.name, img_path.name
                                )
                                st.image(
                                    str(img_path),
                                    caption=description,
                                    width="stretch",
                                )

                    # Display sources
                    if results:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("### 📚 Sources Used")

                        for idx, result in enumerate(results, 1):
                            filename = Path(result["source"]).name
                            badges = [
                                f"📄 Page {result['page']}",
                                f"📊 Relevance: {(1 / (1 + result['score'])):.2%}",
                            ]
                            badge_html = " • ".join(badges)
                            expander_label = f"Source {idx}: {badge_html}"

                            with st.expander(expander_label):
                                st.markdown(result["text"])

            except Exception as e:
                st.error(f"❌ Error during query: {str(e)}")

    # ============================================================================
    # CHART BROWSER
    # ============================================================================

    chart_files = get_all_chart_images(st.session_state.chart_dir)

    if chart_files:
        # Initialize chart browser index
        if "chart_browser_idx" not in st.session_state:
            st.session_state.chart_browser_idx = 0

        # Reset index if files changed
        if (
            "cached_chart_files" not in st.session_state
            or st.session_state.cached_chart_files != chart_files
        ):
            st.session_state.cached_chart_files = chart_files
            st.session_state.chart_browser_idx = 0

        total = len(chart_files)
        idx = st.session_state.chart_browser_idx

        st.divider()
        with st.expander(
            "### 📊 Detected Charts and Descriptions (Entire Doc)", expanded=False
        ):
            st.markdown(
                "<p style='color: #b0b0ff;'>Charts extracted from your document</p>",
                unsafe_allow_html=True,
            )

            # Current chart
            chart_path = chart_files[idx]
            description = st.session_state.rag_pipeline.chart_descriptions.get(
                chart_path.name, chart_path.name
            )
            page_number = extract_page_number(chart_path)

            # Navigation
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.button("⬅️ Previous", disabled=(total <= 1)):
                    st.session_state.chart_browser_idx = (idx - 1) % total
                    st.rerun()
            with col2:
                st.markdown(
                    f"<center>{idx + 1} / {total}</center>", unsafe_allow_html=True
                )
                st.markdown(f"#### Page {page_number}")
                st.image(str(chart_path), width="stretch")
                st.html(f"<small>{description}</small>")
            with col3:
                if st.button("Next ➡️", disabled=(total <= 1)):
                    st.session_state.chart_browser_idx = (idx + 1) % total
                    st.rerun()
