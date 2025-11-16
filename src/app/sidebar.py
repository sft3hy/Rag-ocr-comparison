import streamlit as st
import os
import time
import tempfile
from pathlib import Path
import traceback

# Import from the new project structure
from src.core.rag_pipeline import SmartRAG
from src.vision.vision_models import VisionModelFactory
from .ui_utils import get_chart_output_dir, get_all_chart_images


def display_sidebar():
    """
    Renders the entire sidebar, including model selection, file uploading,
    and informational sections.
    """
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)

        # Check for API key
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            st.warning(
                "GROQ_API_KEY environment variable not set. Please add it to proceed."
            )
            st.stop()

        display_vision_model_selection()
        display_document_uploader(groq_api_key)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")
        display_technology_explanations()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        display_supported_formats_and_footer()


def display_vision_model_selection():
    """
    Handles the UI for selecting the vision model and displaying its information.
    """
    st.markdown("### 🤖 Vision Model Selection")

    available_models = VisionModelFactory.get_available_models()
    model_descriptions = {
        "Moondream2": "Fast & compact (1.6B) - Best for speed",
        "Qwen3-VL-2B": "Balanced performance (2B) - Good accuracy",
        "InternVL3.5-1B": "High accuracy (1B) - Best for quality",
    }

    selected_model = st.selectbox(
        "Choose a vision model",
        available_models,
        index=available_models.index(st.session_state.selected_vision_model),
        format_func=lambda x: f"{x} - {model_descriptions.get(x, '')}",
        help="Select which vision model to use for analyzing charts and images. Runs locally.",
    )

    # Update session state and reset processing if the model is changed
    if selected_model != st.session_state.selected_vision_model:
        st.session_state.selected_vision_model = selected_model
        if st.session_state.processing_complete:
            st.warning("⚠️ Vision model changed. Please re-process your document.")
            st.session_state.processing_complete = False
            st.session_state.rag_pipeline = None

    # st.markdown(f"**Selected:** `{selected_model}`")

    with st.expander("ℹ️ Model Information"):
        if selected_model == "Moondream2":
            st.markdown(
                """
                - **Performance:** ~4 seconds per image
                - **Parameters:** 1.6 Billion
                - **Speed:** ⚡⚡⚡ Very Fast
                - **Accuracy:** ⭐⭐ Good
                - **Use Case:** Ideal for quick analysis and rapid prototyping.
                - **Memory:** ~7GB VRAM
                """
            )
        elif selected_model == "Qwen3-VL-2B":
            st.markdown(
                """
                - **Performance:** ~40 seconds per image
                - **Parameters:** 2 Billion
                - **Speed:** ⚡⚡ Fast
                - **Accuracy:** ⭐⭐⭐ Very Good
                - **Use Case:** A strong balance between speed and detailed analysis.
                - **Memory:** ~12GB VRAM
                """
            )
        elif selected_model == "InternVL3.5-1B":
            st.markdown(
                """
                - **Performance:** ~28 seconds per image
                - **Parameters:** 1 Billion
                - **Speed:** ⚡ Moderate
                - **Accuracy:** ⭐⭐⭐⭐ Excellent
                - **Use Case:** Best choice for high-stakes analysis requiring top accuracy.
                - **Memory:** ~12GB VRAM
                """
            )


def display_document_uploader(groq_api_key: str):
    """
    Handles the file uploader and the 'Process Document' button.

    Args:
        groq_api_key (str): The Groq API key. Processing is disabled if this is not present.
    """
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF, DOCX, or PPTX file",
        type=["pdf", "docx", "pptx"],
        label_visibility="collapsed",
    )

    if uploaded_file and groq_api_key:
        if st.button("🚀 Process Document", use_container_width=True):
            process_document(uploaded_file)


def process_document(uploaded_file):
    """
    Handles the core logic of processing the uploaded document.
    """
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()

    try:
        # Create a unique directory for chart outputs for this run
        chart_dir = get_chart_output_dir(uploaded_file.name)
        chart_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.chart_dir = chart_dir.resolve()

        # Save uploaded file to a temporary path
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.temp_file_path = tmp_file.name

        status_text.markdown("✅ **Temporary file created**")
        progress_bar.progress(5, text="File saved. Initializing models.")

        # Initialize RAG pipeline with the selected vision model
        status_text.markdown(
            f"🧠 **Initializing SmartRAG with `{st.session_state.selected_vision_model}`...**"
        )
        st.session_state.rag_pipeline = SmartRAG(
            output_dir=str(st.session_state.chart_dir),
            vision_model_name=st.session_state.selected_vision_model,
        )

        progress_bar.progress(
            20, text="Models initialized. Starting document analysis."
        )
        status_text.markdown("📊 **Parsing document and detecting charts...**")

        # Define a callback function to update the progress bar from the pipeline
        def update_progress(current, total):
            progress_percent = 20 + int((current / total) * 70)
            progress_bar.progress(
                progress_percent, text=f"Processing page {current}/{total}..."
            )

        # Run the main indexing process
        st.session_state.rag_pipeline.index_document(
            st.session_state.temp_file_path,
            chunk_size=500,
            overlap=100,
            progress_callback=update_progress,
        )

        progress_bar.progress(95, text="Building search index...")
        time.sleep(0.5)

        progress_bar.progress(100, text="Processing Complete!")
        status_text.markdown("✅ **Document processing complete!**")
        st.session_state.processing_complete = True
        time.sleep(1)

        # Final status update
        actual_charts = get_all_chart_images(st.session_state.chart_dir)
        if actual_charts:
            status_text.success(f"📊 Found and analyzed {len(actual_charts)} charts.")
        else:
            status_text.info("⚠️ No charts were detected in the document.")

        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        st.success(f"Document processed with {st.session_state.selected_vision_model}!")

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ An error occurred during processing: {str(e)}")
        traceback.print_exc()  # Log the full error to the console for debugging
    finally:
        # Clean up the temporary file
        if st.session_state.temp_file_path and os.path.exists(
            st.session_state.temp_file_path
        ):
            os.remove(st.session_state.temp_file_path)
            st.session_state.temp_file_path = None


def display_technology_explanations():
    """Renders the expanders explaining the technologies used."""
    st.markdown("### 🔧 Technologies Used")

    with st.expander("🎯 Combined Chart Detection"):
        st.markdown(
            """
            This app uses a **dual-model system** to maximize chart detection accuracy:
            - **Table Transformer (TATR):** An AI model excellent for finding structured tables and charts.
            - **Heuristic Detection:** A rules-based algorithm that catches charts TATR might miss by analyzing pixel variance and text density.
            
            The results are combined and de-duplicated to ensure comprehensive coverage.
            """
        )
    with st.expander("👁️ Local Vision Models"):
        st.markdown(
            """
            Chart and image understanding is performed **locally on your machine** using one of three powerful open-source vision models. By analyzing images locally, your data remains private. You can choose the model that best fits your hardware and accuracy needs.
            """
        )
    with st.expander("🤖 Groq + Llama 4 for Q&A"):
        st.markdown(
            """
            Question-answering is powered by **Llama 4 (17B)** running on the **Groq LPU™ Inference Engine**. This provides extremely fast and accurate language understanding, allowing for near-instantaneous answers based on your document's content.
            """
        )


def display_supported_formats_and_footer():
    """Renders the supported formats list and the sidebar footer."""
    st.markdown("### Supported Formats")
    st.markdown(
        """
            - 📕 PDF Documents (.pdf)
            - 📝 Word Documents (.docx)
            - 📊 PowerPoint (.pptx)
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #fff; font-size: 0.8rem;'>Built with Streamlit</p>",
        unsafe_allow_html=True,
    )
