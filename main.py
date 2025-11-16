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
        page_title="Smart RAG Document Analyzer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

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

    # Display UI components
    display_header()
    display_sidebar()
    display_main_content()


if __name__ == "__main__":
    main()
