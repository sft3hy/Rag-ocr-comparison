import streamlit as st
from rag_pipeline import RAGPipeline
import os
import time

from custom_css import custom

# Page config with custom theme
st.set_page_config(
    page_title="RAG OCR Comparison Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, professional look with animations
st.markdown(
    custom,
    unsafe_allow_html=True,
)

# Initialize session state
if "pipeline_no_ocr" not in st.session_state:
    st.session_state.pipeline_no_ocr = None
if "pipeline_with_ocr" not in st.session_state:
    st.session_state.pipeline_with_ocr = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# Header with icon and subtitle
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        "<h1 style='text-align: center;'>🔬 RAG OCR Comparison Lab</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #e0e0ff; font-size: 1.2rem; margin-top: -1rem;'>Analyze document retrieval performance with and without OCR</p>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # st.markdown("### ⚙️ Configuration")
    st.markdown("<br>", unsafe_allow_html=True)

    groq_api_key = os.environ.get("GROQ_API_KEY", "")

    # if groq_api_key:
    #     st.markdown("✅ **API Key Loaded**")
    # else:
    #     st.error("❌ GROQ_API_KEY environment variable not set!")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "png", "jpg", "jpeg", "pptx", "docx"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if uploaded_file and groq_api_key:
        if st.button("🚀 Process Document", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process without OCR
            status_text.markdown("🔄 **Processing without OCR...**")
            progress_bar.progress(25)
            time.sleep(0.3)

            st.session_state.pipeline_no_ocr = RAGPipeline(groq_api_key, use_ocr=False)
            st.session_state.pipeline_no_ocr.process_document(uploaded_file)

            progress_bar.progress(50)
            status_text.markdown(
                "🔄 **Processing with OCR...**",
                unsafe_allow_html=True,
            )
            time.sleep(0.3)

            # Reset file pointer
            uploaded_file.seek(0)

            # Process with OCR
            st.session_state.pipeline_with_ocr = RAGPipeline(groq_api_key, use_ocr=True)
            st.session_state.pipeline_with_ocr.process_document(uploaded_file)

            progress_bar.progress(100)
            status_text.markdown("✅ **Processing complete!**")
            st.session_state.processing_complete = True
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            st.markdown(
                "<div class='success-message'>✨ Document processed successfully!</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Supported Formats")
    st.markdown(
        """<div class='sidebarText'>
    - 📕 PDF Documents<br>
    - 📘 Word Documents (.docx)<br>
    - 📙 PowerPoint (.pptx)<br>
    - 🖼️ Images (PNG, JPG)<br>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888; font-size: 0.8rem;'>Built with Streamlit • Powered by Groq</p>",
        unsafe_allow_html=True,
    )

# Main content area
if not st.session_state.processing_complete:
    # Welcome screen
    st.markdown(
        """
    <div style='text-align: center; padding: 4rem 2rem; background: rgba(255, 255, 255, 0.05); border-radius: 20px; backdrop-filter: blur(10px);'>
        <h2 style='font-size: 2rem; margin-bottom: 1rem;'>👋 Welcome to the RAG OCR Comparison Lab</h2>
        <p style='font-size: 1.2rem; color: #e0e0ff; margin-bottom: 2rem;'>
            Upload a document to compare retrieval performance with and without OCR
        </p>
        <div style='display: flex; justify-content: center; gap: 3rem; margin-top: 3rem; flex-wrap: wrap;'>
            <div class='metric-card' style='min-width: 200px;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>📊</div>
                <div style='font-weight: 600; font-size: 1.1rem;'>Side-by-side Comparison</div>
            </div>
            <div class='metric-card' style='min-width: 200px;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>⚡</div>
                <div style='font-weight: 600; font-size: 1.1rem;'>Real-time Processing</div>
            </div>
            <div class='metric-card' style='min-width: 200px;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🎯</div>
                <div style='font-weight: 600; font-size: 1.1rem;'>Accurate Results</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
else:
    # Create tabs
    tab1, tab2 = st.tabs(["📄 Without OCR", "🔍 With OCR"])

    with tab1:
        st.markdown("### RAG Pipeline without OCR")
        st.markdown(
            "<p style='color: #e0e0ff; margin-bottom: 2rem;'>Direct text extraction from documents</p>",
            unsafe_allow_html=True,
        )

        if st.session_state.pipeline_no_ocr is None:
            st.info("📤 Please upload and process a document first")
        else:
            # Stats row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9rem; color: #b0b0ff; margin-bottom: 0.5rem;'>TOTAL CHARACTERS</div>
                    <div style='font-size: 2rem; font-weight: 700;'>{len(st.session_state.pipeline_no_ocr.full_text):,}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9rem; color: #b0b0ff; margin-bottom: 0.5rem;'>TEXT CHUNKS</div>
                    <div style='font-size: 2rem; font-weight: 700;'>{len(st.session_state.pipeline_no_ocr.chunks)}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                words = len(st.session_state.pipeline_no_ocr.full_text.split())
                st.markdown(
                    f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9rem; color: #b0b0ff; margin-bottom: 0.5rem;'>WORD COUNT</div>
                    <div style='font-size: 2rem; font-weight: 700;'>{words:,}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Show extracted text preview
            with st.expander("👁️ View Extracted Text"):
                st.markdown(
                    st.session_state.pipeline_no_ocr.full_text,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Question input
            question = st.text_input(
                "💬 Ask a question about your document:",
                key="q1",
                placeholder="e.g., What is the main topic of this document?",
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                ask_button = st.button(
                    "🎯 Get Answer", key="btn1", use_container_width=True
                )

            if ask_button and question:
                with st.spinner("🤔 Thinking..."):
                    answer = st.session_state.pipeline_no_ocr.query(question)

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

    with tab2:
        st.markdown("### RAG Pipeline with OCR")
        st.markdown(
            "<p style='color: #e0e0ff; margin-bottom: 2rem;'>Optical Character Recognition enabled</p>",
            unsafe_allow_html=True,
        )

        if st.session_state.pipeline_with_ocr is None:
            st.info("📤 Please upload and process a document first")
        else:
            # Stats row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9rem; color: #b0b0ff; margin-bottom: 0.5rem;'>TOTAL CHARACTERS</div>
                    <div style='font-size: 2rem; font-weight: 700;'>{len(st.session_state.pipeline_with_ocr.full_text):,}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9rem; color: #b0b0ff; margin-bottom: 0.5rem;'>TEXT CHUNKS</div>
                    <div style='font-size: 2rem; font-weight: 700;'>{len(st.session_state.pipeline_with_ocr.chunks)}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                words = len(st.session_state.pipeline_with_ocr.full_text.split())
                st.markdown(
                    f"""
                <div class='metric-card'>
                    <div style='font-size: 0.9rem; color: #b0b0ff; margin-bottom: 0.5rem;'>WORD COUNT</div>
                    <div style='font-size: 2rem; font-weight: 700;'>{words:,}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Show extracted text preview
            with st.expander("👁️ View Extracted Text (OCR)"):
                st.markdown(
                    st.session_state.pipeline_with_ocr.full_text,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Question input
            question = st.text_input(
                "💬 Ask a question about your document:",
                key="q2",
                placeholder="e.g., What is the main topic of this document?",
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                ask_button = st.button(
                    "🎯 Get Answer", key="btn2", use_container_width=True
                )

            if ask_button and question:
                with st.spinner("🤔 Thinking..."):
                    answer = st.session_state.pipeline_with_ocr.query(question)

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
