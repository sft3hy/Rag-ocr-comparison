import streamlit as st


def display_header():
    """Displays the main header of the application."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<h1 style='text-align: center;'>🧠 Smart RAG Document Analyzer</h1>",
            unsafe_allow_html=True,
        )
        # st.markdown(
        #     "<p style='text-align: center; color: #e0e0ff; font-size: 1.2rem; margin-top: -1rem;'>AI-powered document analysis with OCR and vision understanding</p>",
        #     unsafe_allow_html=True,
        # )
    st.markdown("<br>", unsafe_allow_html=True)
