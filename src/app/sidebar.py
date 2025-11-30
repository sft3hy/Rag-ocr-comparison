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
        if not st.session_state.get("active_document_ids"):
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

    # Get list of past sessions from the database
    past_sessions = st.session_state.db_manager.get_all_sessions()

    # Create options for the selectbox
    # Format: "session_name (timestamp) - X docs" -> session_id
    session_options = {}
    for session_id, session_name, timestamp, doc_count in past_sessions:
        doc_text = "doc" if doc_count == 1 else "docs"
        display_name = f"{session_name} ({timestamp}) - {doc_count} {doc_text}"
        session_options[display_name] = session_id

    options_list = ["✨ Start New Session"] + list(session_options.keys())

    selected_option = st.selectbox("Choose a previous session", options=options_list)

    if selected_option != "✨ Start New Session":
        if st.button("Load Selected Session", width="stretch"):
            session_id_to_load = session_options[selected_option]
            load_session(session_id_to_load)
    else:
        if st.button("Start New Session", width="stretch"):
            # Reset all relevant session state variables
            st.session_state.processing_complete = False
            st.session_state.rag_pipelines = []
            st.session_state.active_document_ids = []
            st.session_state.active_session_id = None
            st.session_state.chat_history = []
            st.rerun()


def load_session(session_id: int):
    """Handles the logic of loading a past session from the database."""
    with st.spinner(f"Loading session {session_id}..."):
        try:
            # Get all documents in this session
            documents = st.session_state.db_manager.get_session_documents(session_id)
            if not documents:
                st.error("Could not find session data.")
                return

            # Load all RAG pipelines for the documents in this session
            rag_pipelines = []
            doc_ids = []

            for doc_data in documents:
                # Re-initialize the RAG pipeline with the saved vision model
                rag_pipeline = SmartRAG(
                    output_dir=doc_data["chart_dir"],
                    vision_model_name=doc_data["vision_model_used"],
                    load_vision=False,
                )

                # Load the saved state (FAISS index and chunks)
                rag_pipeline.load_state(
                    doc_data["faiss_index_path"], doc_data["chunks_path"]
                )
                rag_pipeline.chart_descriptions = doc_data["chart_descriptions"]

                rag_pipelines.append(rag_pipeline)
                doc_ids.append(doc_data["id"])

            # Restore state with all documents
            st.session_state.rag_pipelines = rag_pipelines
            st.session_state.processing_complete = True
            st.session_state.active_document_ids = doc_ids
            st.session_state.active_session_id = session_id
            st.session_state.selected_vision_model = documents[0]["vision_model_used"]

            # Load chat history for this session
            st.session_state.chat_history = (
                st.session_state.db_manager.get_queries_for_session(session_id)
            )

            doc_names = [doc["original_filename"] for doc in documents]
            if len(doc_names) == 1:
                st.success(f"Successfully loaded session with '{doc_names[0]}'.")
            else:
                st.success(
                    f"Successfully loaded session with {len(doc_names)} documents."
                )

            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"Failed to load session: {e}")
            traceback.print_exc()


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
                load_vision=False,
            )

            # Load the saved state (FAISS index and chunks)
            rag_pipeline.load_state(
                doc_data["faiss_index_path"], doc_data["chunks_path"]
            )
            rag_pipeline.chart_descriptions = doc_data["chart_descriptions"]

            # Restore state with lists
            st.session_state.rag_pipelines = [rag_pipeline]
            st.session_state.processing_complete = True
            st.session_state.active_document_ids = [doc_id]
            st.session_state.selected_vision_model = doc_data["vision_model_used"]

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
            st.session_state.rag_pipelines = []

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

    uploaded_files = st.file_uploader(
        "Choose a PDF, DOCX, or PPTX file",
        type=["pdf", "docx", "pptx"],
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if uploaded_files and groq_api_key:
        button_text = "🚀 Process Document"
        if len(uploaded_files) > 1:
            button_text += "s"
        if st.button(button_text, width="stretch"):
            process_document(uploaded_files)


def process_document(uploaded_files):
    """
    Handles the core logic of processing multiple uploaded documents.
    """
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()

    # Store document IDs and pipelines for all processed files
    processed_doc_ids = []
    processed_pipelines = []

    try:
        # Create a new session for these documents
        filenames = [f.name for f in uploaded_files]
        session_id = st.session_state.db_manager.create_session(filenames)
        st.session_state.active_session_id = session_id  # Set immediately

        total_files = len(uploaded_files)

        for file_index, uploaded_file in enumerate(uploaded_files):
            # Update overall progress
            file_progress_start = int((file_index / total_files) * 100)
            file_progress_range = int(100 / total_files)

            status_text.markdown(
                f"📄 **Processing file {file_index + 1}/{total_files}: {uploaded_file.name}**"
            )

            # Create a unique directory for chart outputs for this file
            chart_dir = get_chart_output_dir(uploaded_file.name)
            chart_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded file to a temporary path
            file_ext = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name

            status_text.markdown(
                f"✅ **Temporary file created for {uploaded_file.name}**"
            )
            progress_bar.progress(
                file_progress_start + int(file_progress_range * 0.05),
                text=f"File {file_index + 1}/{total_files} saved. Initializing models.",
            )

            # Initialize RAG pipeline with the selected vision model
            status_text.markdown(
                f"🧠 **Initializing SmartRAG for {uploaded_file.name} with `{st.session_state.selected_vision_model}`...**"
            )
            rag_pipeline = SmartRAG(
                output_dir=str(chart_dir),
                vision_model_name=st.session_state.selected_vision_model,
            )

            progress_bar.progress(
                file_progress_start + int(file_progress_range * 0.20),
                text=f"Models initialized for file {file_index + 1}/{total_files}. Starting document analysis.",
            )
            status_text.markdown(
                f"📊 **Parsing {uploaded_file.name} and detecting charts...**"
            )

            # Define a callback function to update the progress bar from the pipeline
            def update_progress(current, total):
                progress_percent = (
                    file_progress_start
                    + int(file_progress_range * 0.20)
                    + int((current / total) * file_progress_range * 0.70)
                )
                progress_bar.progress(
                    progress_percent,
                    text=f"Processing file {file_index + 1}/{total_files} - page {current}/{total}...",
                )

            # Run the main indexing process
            rag_pipeline.index_document(
                temp_file_path,
                chunk_size=500,
                overlap=100,
                progress_callback=update_progress,
            )

            progress_bar.progress(
                file_progress_start + int(file_progress_range * 0.95),
                text=f"Saving processed state for file {file_index + 1}/{total_files}...",
            )

            # Create the DB record for this document, linked to the session
            doc_id = st.session_state.db_manager.add_document_record(
                filename=uploaded_file.name,
                vision_model=st.session_state.selected_vision_model,
                chart_dir=str(chart_dir),
                faiss_path="",  # Placeholder
                chunks_path="",  # Placeholder
                chart_descriptions=rag_pipeline.chart_descriptions,
                session_id=session_id,  # Link to session
            )
            processed_doc_ids.append(doc_id)
            processed_pipelines.append(rag_pipeline)

            # Save the FAISS index and chunks using the new doc_id
            faiss_path, chunks_path = rag_pipeline.save_state(doc_id)

            # Update the record with the correct paths
            st.session_state.db_manager.conn.execute(
                "UPDATE documents SET faiss_index_path = ?, chunks_path = ? WHERE id = ?",
                (faiss_path, chunks_path, doc_id),
            )
            st.session_state.db_manager.conn.commit()

            # Clean up the temporary file for this document
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            # Report charts found for this file
            actual_charts = get_all_chart_images(chart_dir)
            if actual_charts:
                status_text.markdown(
                    f"✅ **{uploaded_file.name}**: Found and analyzed {len(actual_charts)} charts."
                )
            else:
                status_text.markdown(f"⚠️ **{uploaded_file.name}**: No charts detected.")

        # All files processed
        progress_bar.progress(100, text="All documents processed!")
        status_text.markdown(
            f"✅ **Successfully processed {total_files} document(s)!**"
        )

        # Set all processed documents as active
        st.session_state.active_document_ids = processed_doc_ids
        st.session_state.rag_pipelines = processed_pipelines
        st.session_state.chat_history = []  # Initialize empty chat history
        st.session_state.processing_complete = True

        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

        st.success(
            f"✅ Processed {total_files} document(s) with {st.session_state.selected_vision_model}!"
        )

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ An error occurred during processing: {str(e)}")
        traceback.print_exc()  # Log the full error to the console for debugging


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
