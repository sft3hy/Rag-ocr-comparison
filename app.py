import streamlit as st
from rag_enhanced import SmartRAG
from vision_models import VisionModelFactory
import os
import time
import tempfile
from pathlib import Path
from PIL import Image

from custom_css import custom

# Page config
st.set_page_config(
    page_title="Smart RAG Document Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(custom, unsafe_allow_html=True)

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None
if "chart_dir" not in st.session_state:
    st.session_state.chart_dir = None
if "selected_vision_model" not in st.session_state:
    st.session_state.selected_vision_model = "Moondream2"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_chart_output_dir(filename: str) -> Path:
    """Create a unique output directory path for charts."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = Path(filename).stem.replace(" ", "_")
    dir_name = f"{timestamp}_combined_{safe_name}"
    return Path("potential_charts") / dir_name


def get_all_chart_images(chart_dir: Path | None) -> list[Path]:
    """Get all chart PNG files from the directory, sorted by page number."""
    if chart_dir is None or not chart_dir.exists():
        return []
    chart_files = list(chart_dir.glob("*.png"))
    return sorted(chart_files, key=extract_page_number)


def extract_page_number(filepath: Path) -> int:
    """Extract page/slide number from filename."""
    import re

    match = re.search(r"page(\d+)", filepath.name)
    if match:
        return int(match.group(1))
    match = re.search(r"slide(\d+)", filepath.name)
    if match:
        return int(match.group(1))
    return float("inf")


def get_charts_for_page(chart_dir: Path, page: int) -> list[Path]:
    """Get all chart images for a specific page."""
    if not chart_dir or not chart_dir.exists():
        return []
    patterns = [f"page{page}_chart*.png", f"page{page}_embedded*.png"]
    charts = []
    for pattern in patterns:
        charts.extend(chart_dir.glob(pattern))
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

    # NEW: Vision Model Selection
    st.markdown("### 🤖 Vision Model Selection")

    available_models = VisionModelFactory.get_available_models()

    # Model descriptions
    model_descriptions = {
        "Moondream2": "Fast & compact (1.6B params) - Best for speed",
        "Qwen3-VL-2B": "Balanced performance (2B params) - Good accuracy",
        "InternVL3.5-1B": "High accuracy (1B params) - Best for quality",
    }

    selected_model = st.selectbox(
        "Choose a vision model",
        available_models,
        index=available_models.index(st.session_state.selected_vision_model),
        format_func=lambda x: f"{x} - {model_descriptions.get(x, '')}",
        help="Select which vision model to use for analyzing charts and images",
    )

    # Update session state if model changed
    if selected_model != st.session_state.selected_vision_model:
        st.session_state.selected_vision_model = selected_model
        # Reset processing state if model changes
        if st.session_state.processing_complete:
            st.warning("⚠️ Vision model changed. Please re-process your document.")
            st.session_state.processing_complete = False
            st.session_state.rag_pipeline = None

    st.markdown(f"**Selected:** {selected_model}")

    # Show model info
    with st.expander("ℹ️ Model Information"):
        if selected_model == "Moondream2":
            st.markdown(
                """
            **Moondream2**
            - Performance: ~4 seconds per image description
            - Parameters: ~1.6B
            - Speed: ⚡⚡⚡ Very Fast
            - Accuracy: ⭐⭐ Good
            - Best for: Quick processing
            - Memory: ~7GB
            """
            )
        elif selected_model == "Qwen3-VL-2B":
            st.markdown(
                """
            **Qwen3-VL-2B**
            - Performance: ~40 seconds per image description
            - Parameters: 2B
            - Speed: ⚡⚡ Fast
            - Accuracy: ⭐⭐⭐ Very Good
            - Best for: Balanced use
            - Memory: ~12GB
            """
            )
        elif selected_model == "InternVL3.5-1B":
            st.markdown(
                """
            **InternVL3.5-1B**
            - Performance: ~ 28 seconds per image description
            - Parameters: 1B
            - Speed: ⚡ Moderate
            - Accuracy: ⭐⭐⭐⭐ Excellent
            - Best for: High accuracy
            - Memory: ~12GB
            """
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "pptx"],
        label_visibility="collapsed",
    )

    if uploaded_file and groq_api_key:
        if st.button("🚀 Process Document", width="stretch"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                chart_dir = get_chart_output_dir(uploaded_file.name)
                chart_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.chart_dir = chart_dir.resolve()

                file_ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_ext
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    st.session_state.temp_file_path = tmp_file.name

                progress_bar.progress(10)
                status_text.markdown(f"🔄 **Loading {selected_model} vision model...**")
                time.sleep(0.3)

                progress_bar.progress(25)
                status_text.markdown(
                    "📊 **Starting document analysis with combined detection...**"
                )
                time.sleep(0.3)

                # Initialize RAG with selected vision model
                st.session_state.rag_pipeline = SmartRAG(
                    output_dir=str(st.session_state.chart_dir),
                    vision_model_name=selected_model,  # Pass the selected model
                )

                def update_progress(current_page, total_pages):
                    progress = 25 + int((current_page / total_pages) * 65)
                    progress_bar.progress(progress)
                    status_text.markdown(
                        f"📊 **Processing page {current_page}/{total_pages} with {selected_model}...**"
                    )
                    time.sleep(0.1)

                st.session_state.rag_pipeline.index_document(
                    st.session_state.temp_file_path,
                    chunk_size=500,
                    overlap=100,
                    progress_callback=update_progress,
                )

                progress_bar.progress(90)
                status_text.markdown("🔍 **Building search index...**")
                time.sleep(0.3)

                progress_bar.progress(100)
                status_text.markdown("✅ **Processing complete!**")
                st.session_state.processing_complete = True

                actual_charts = get_all_chart_images(st.session_state.chart_dir)
                time.sleep(0.5)

                if actual_charts:
                    status_text.markdown(
                        f"📊 Found {len(actual_charts)} charts (Combined TATR + Heuristic)"
                    )
                else:
                    status_text.markdown("⚠️ No charts detected in document")

                time.sleep(2)
                progress_bar.empty()
                status_text.empty()

                st.markdown(
                    f"<div class='success-message'>✨ Document processed successfully with {selected_model}!</div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error processing document: {str(e)}")
                import traceback

                print(traceback.format_exc())

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    # Technology explanations
    st.markdown("### 🔧 Technologies Used")

    with st.expander("📷 Tesseract OCR"):
        st.markdown(
            """
        **Optical Character Recognition Engine**
        
        Extracts text from images and scanned documents.
        
        **In this app:**
        - Extracts text from charts and images
        - Handles low-quality scans
        - Provides confidence scores
        """
        )

    # NEW: Updated vision model expander
    with st.expander("👁️ Vision Models (Switchable)"):
        st.markdown(
            """
        **Multiple AI Vision Models Available**
        
        Choose from three different vision models:
        
        **1. Moondream2:**
        - Fastest processing speed
        - Compact 1.6B parameter model
        - Good for quick analysis
        - Around 4 seconds per image
        
        **2. Qwen3-VL-2B:**
        - Balanced speed and accuracy
        - 2B parameters
        - Excellent chart understanding
        - Around 40 seconds per image
        
        **3. InternVL3.5-1B:**
        - Best accuracy
        - 1B parameters
        - Superior for complex visuals
        - Around 28 seconds per image
        
        **All models:**
        - Analyze charts, graphs, and diagrams
        - Extract trends and data points
        - Generate detailed descriptions
        - Run locally (MPS/CUDA/CPU)
        """
        )

    with st.expander("🎯 Combined Chart Detection"):
        st.markdown(
            """
        **Dual-Model Detection System**

        **1. Table Transformer (TATR):**
        - AI-powered transformer model
        - Excellent for structured content

        **2. Heuristic Detection:**
        - Grid-based analysis
        - Catches edge cases

        **Combined Approach:**
        - Runs both detectors
        - Removes duplicates via IoU
        - Maximum chart coverage
        """
        )

    with st.expander("🤖 Groq + Llama 4"):
        st.markdown(
            """
        **Fast Language Model Inference**
        
        - Ultra-fast inference
        - Context-aware responses
        - 750+ tokens/second
        """
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Supported Formats")
    st.markdown(
        """<div class='sidebarText'>
    - 📕 PDF Documents<br>
    - 📝 Word Documents (.docx)<br>
    - 📊 PowerPoint (.pptx)<br>
    - 🖼️ Scanned PDFs<br>
    - 📊 Charts/Graphs<br>
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
        f"""
    <div style='text-align: center; padding: 4rem 2rem; background: rgba(255, 255, 255, 0.05); border-radius: 20px; backdrop-filter: blur(10px);'>
        <h2 style='font-size: 2rem; margin-bottom: 1rem;'>👋 Welcome to Smart RAG</h2>
        <p style='font-size: 1.2rem; color: #e0e0ff; margin-bottom: 2rem;'>
            Upload a document to unlock advanced AI analysis
        </p>
        <p style='font-size: 1rem; color: #b0b0ff; margin-bottom: 2rem;'>
            Currently using: <strong>{st.session_state.selected_vision_model}</strong> vision model
        </p>
        <div style='display: flex; justify-content: center; gap: 3rem; margin-top: 3rem; flex-wrap: wrap;'>
            <div class='metric-card' style='min-width: 200px;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>📊</div>
                <div style='font-weight: 600; font-size: 1.1rem;'>Chart Understanding</div>
                <div style='font-size: 0.9rem; color: #b0b0ff; margin-top: 0.5rem;'>3 Vision Models Available</div>
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
    # Show which model was used
    if st.session_state.rag_pipeline:
        model_name = st.session_state.rag_pipeline.vision_model_name
        # st.info(f"📊 Document processed with **{model_name}** vision model")

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

                    chart_images = []
                    for result in results:
                        if "[CHART DESCRIPTION" in result["text"]:
                            page = result["page"]
                            page_charts = get_charts_for_page(
                                st.session_state.chart_dir, page
                            )
                            chart_images.extend(page_charts)

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

                    if chart_images:
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander(
                            f"🖼️ Image Descriptions Used in Response ({len(chart_images)})"
                        ):
                            for img_path in chart_images:
                                if st.session_state.rag_pipeline.chart_descriptions:
                                    description = st.session_state.rag_pipeline.chart_descriptions.get(
                                        img_path.name, img_path.name
                                    )
                                else:
                                    description = img_path.name
                                cols = st.columns([1, 2, 1])
                                with cols[1]:
                                    st.image(
                                        str(img_path),
                                        caption=description,
                                        width="content",
                                        # width="stretch",
                                    )

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
                import traceback

                print(traceback.format_exc())

    # ============================================================================
    # CHART BROWSER
    # ============================================================================

    chart_files = get_all_chart_images(st.session_state.chart_dir)

    if chart_files:
        if "chart_browser_idx" not in st.session_state:
            st.session_state.chart_browser_idx = 0

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
            f"### 📊 Detected Charts (Combined TATR + Heuristic) - {total} total",
            expanded=True,
        ):
            st.markdown(
                f"<p style='color: #b0b0ff;'>Charts analyzed using <strong>{st.session_state.rag_pipeline.vision_model_name}</strong></p>",
                unsafe_allow_html=True,
            )

            chart_path = chart_files[idx]

            if st.session_state.rag_pipeline.chart_descriptions:
                description = st.session_state.rag_pipeline.chart_descriptions.get(
                    chart_path.name, chart_path.name
                )
            else:
                description = chart_path.name

            page_number = extract_page_number(chart_path)

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
                st.markdown(f"<small>{description}</small>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", disabled=(total <= 1)):
                    st.session_state.chart_browser_idx = (idx + 1) % total
                    st.rerun()
