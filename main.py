import streamlit as st
from static.custom_css import custom_css
from src.app.header import display_header
from src.app.sidebar import display_sidebar
from src.app.main_content import display_main_content
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """
    Main function to run the Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        # page_title="Smart RAG Document Analyzer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Initialize session state variables
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = None

    if "active_document_ids" not in st.session_state:
        st.session_state.active_document_ids = []

    if "rag_pipelines" not in st.session_state:
        st.session_state.rag_pipelines = []

    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "selected_vision_model" not in st.session_state:
        st.session_state.selected_vision_model = "Moondream2"

    # Display UI components
    display_header()
    display_sidebar()
    display_main_content()


if __name__ == "__main__":
    main()
