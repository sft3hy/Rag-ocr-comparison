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
from src.utils.db_utils import DatabaseManager


def display_sidebar():
    """Renders the entire sidebar, including session loading."""
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            st.warning("GROQ_API_KEY not set.")
            st.stop()

        # Initialize DB Manager in session state if it doesn't exist
        if "db_manager" not in st.session_state:
            st.session_state.db_manager = DatabaseManager()

        display_session_loader()

        # Show the uploader only for new sessions
        if st.session_state.get("active_document_id") is None:
            display_vision_model_selection()
            display_document_uploader(groq_api_key)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")
        display_technology_explanations()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        display_supported_formats_and_footer()


def display_session_loader():
    """Displays UI for loading past document processing sessions."""
    st.markdown("### 🗂️ Load Session")

    # Get list of past documents from the database
    past_documents = st.session_state.db_manager.get_all_documents()

    # Create options for the selectbox
    # Format: "filename (timestamp)" -> doc_id
    doc_options = {
        f"{filename} ({ts})": doc_id for doc_id, filename, ts in past_documents
    }
    options_list = ["✨ Start New Session"] + list(doc_options.keys())

    selected_option = st.selectbox("Choose a previous document", options=options_list)

    if selected_option != "✨ Start New Session":
        if st.button("Load Selected Session", use_container_width=True):
            doc_id_to_load = doc_options[selected_option]
            load_session(doc_id_to_load)
    elif st.session_state.get("active_document_id") is not None:
        if st.button("End Current Session", use_container_width=True):
            # Reset all relevant session state variables
            st.session_state.processing_complete = False
            st.session_state.rag_pipeline = None
            st.session_state.active_document_id = None
            st.session_state.chat_history = []
            st.rerun()


def load_session(doc_id: int):
    """Handles the logic of loading a past session from the database."""
    with st.spinner(f"Loading session for document ID {doc_id}..."):
        try:
            doc_data = st.session_state.db_manager.get_document_by_id(doc_id)
            if not doc_data:
                st.error("Could not find session data.")
                return

            # Re-initialize the RAG pipeline with the saved vision model
            rag_pipeline = SmartRAG(
                output_dir=doc_data["chart_dir"],
                vision_model_name=doc_data["vision_model_used"],
            )

            # Load the saved state (FAISS index and chunks)
            rag_pipeline.load_state(
                doc_data["faiss_index_path"], doc_data["chunks_path"]
            )

            # Restore state
            st.session_state.rag_pipeline = rag_pipeline
            st.session_state.processing_complete = True
            st.session_state.active_document_id = doc_id
            st.session_state.chart_dir = Path(doc_data["chart_dir"])
            st.session_state.selected_vision_model = doc_data["vision_model_used"]
            st.session_state.rag_pipeline.chart_descriptions = doc_data[
                "chart_descriptions"
            ]

            st.success(
                f"Successfully loaded session for '{doc_data['original_filename']}'."
            )
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"Failed to load session: {e}")
            traceback.print_exc()


def display_vision_model_selection():
    """
    Handles the UI for selecting the vision model and displaying its information.
    """
    st.markdown("### Vision Model Selection")

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

        progress_bar.progress(95, text="Saving processed state...")

        # A new document was processed, so get a new ID from the DB
        # Note: We need to create the DB record *before* saving state to get the ID
        doc_id = st.session_state.db_manager.add_document_record(
            filename=uploaded_file.name,
            vision_model=st.session_state.selected_vision_model,
            chart_dir=str(st.session_state.chart_dir),
            faiss_path="",  # Placeholder
            chunks_path="",  # Placeholder
            chart_descriptions=st.session_state.rag_pipeline.chart_descriptions,
        )
        st.session_state.active_document_id = doc_id

        # Save the FAISS index and chunks using the new doc_id
        faiss_path, chunks_path = st.session_state.rag_pipeline.save_state(doc_id)

        # Now update the record with the correct paths
        st.session_state.db_manager.conn.execute(
            "UPDATE documents SET faiss_index_path = ?, chunks_path = ? WHERE id = ?",
            (faiss_path, chunks_path, doc_id),
        )
        st.session_state.db_manager.conn.commit()

        progress_bar.progress(100, text="Processing Complete!")

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
