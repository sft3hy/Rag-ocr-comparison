import streamlit as st
from pathlib import Path
import traceback

# Import the UI utility functions from the same app directory
from .ui_utils import get_all_chart_images, extract_page_number, get_charts_for_page


def _display_interaction_details(results: list):
    """
    A reusable helper function to display the sources and related charts for a query.

    Args:
        results (list): The list of source dictionaries from a RAG query result.
    """
    if not results:
        return

    with st.expander("📚 View Sources & Related Charts"):
        # Step 1: Find all unique charts related to the source documents
        chart_images = []

        # Iterate through all active pipelines to find charts
        if st.session_state.get("rag_pipelines"):
            for pipeline in st.session_state.rag_pipelines:
                for result in results:
                    if (
                        "[CHART DESCRIPTION" in result["text"]
                        or "[SLIDE VISUAL DESCRIPTION" in result["text"]
                    ):
                        page = result.get("page", 0)
                        # Get chart directory for this pipeline
                        chart_dir = Path(pipeline.output_dir)
                        if chart_dir.exists():
                            page_charts = get_charts_for_page(chart_dir, page)
                            chart_images.extend(page_charts)

        # Remove duplicates while preserving order
        unique_chart_images = list(dict.fromkeys(chart_images))

        # Step 2: Display the related charts
        if unique_chart_images:
            st.markdown("##### 🖼️ Related Charts")
            # Use columns for a cleaner layout if there are few charts
            cols = st.columns(min(len(unique_chart_images), 3))
            for i, img_path in enumerate(unique_chart_images):
                with cols[i % 3]:
                    # Try to find description from any pipeline
                    description = "Chart from document"
                    for pipeline in st.session_state.rag_pipelines:
                        if img_path.name in pipeline.chart_descriptions:
                            description = pipeline.chart_descriptions[img_path.name]
                            break
                    st.image(str(img_path), caption=description)
            st.divider()

        # Step 3: Display the text sources
        st.markdown("##### 📄 Text Sources")
        for idx, result in enumerate(results, 1):
            try:
                # Calculate relevance score, handling potential score format issues
                relevance_score = (1 / (1 + float(result.get("score", 1)))) * 100
                source_info = f"**Source {idx} (from {result.get('source', 'Unknown')} - Page {result.get('page', 'N/A')})** - Relevance: {relevance_score:.1f}%"
            except (ValueError, TypeError):
                source_info = f"**Source {idx} (from {result.get('source', 'Unknown')} - Page {result.get('page', 'N/A')})**"

            with st.container(border=True):
                st.write(source_info)
                st.markdown(result["text"])


def display_main_content():
    """Controls the display of the main content area."""
    if not st.session_state.get("processing_complete", False):
        display_welcome_screen()
    else:
        # Display info about active session
        session_id = st.session_state.get("active_session_id")
        if session_id:
            session_info = st.session_state.db_manager.get_session_info(session_id)
            if session_info:
                doc_count = len(session_info["documents"])
                query_count = session_info["query_count"]

                if doc_count == 1:
                    doc_name = session_info["documents"][0]["original_filename"]
                    st.info(
                        f"📄 Active Document: **{doc_name}** | 💬 {query_count} queries"
                    )
                else:
                    doc_names = [
                        doc["original_filename"] for doc in session_info["documents"]
                    ]
                    st.info(
                        f"📚 Active Session: **{session_info['session_name']}** | 📄 {doc_count} documents | 💬 {query_count} queries"
                    )

                    # Show expandable list of documents
                    with st.expander("📋 View all documents in this session"):
                        for i, doc in enumerate(session_info["documents"], 1):
                            st.markdown(
                                f"{i}. **{doc['original_filename']}** (processed with {doc['vision_model_used']})"
                            )

        display_chat_history()
        display_qa_interface()
        display_chart_browser()


def display_chat_history():
    """Loads and displays the full Q&A history for the active session."""
    session_id = st.session_state.get("active_session_id")
    if session_id:
        # Load history from DB if not already in session state
        if (
            "chat_history" not in st.session_state
            or st.session_state.get("history_session_id") != session_id
        ):
            st.session_state.chat_history = (
                st.session_state.db_manager.get_queries_for_session(session_id)
            )
            st.session_state.history_session_id = session_id

        # Display each interaction from the history
        for interaction in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(interaction["question"])
            with st.chat_message("assistant"):
                st.markdown(interaction["response"])
                # Display the saved sources and charts for this historical query
                _display_interaction_details(interaction.get("sources", []))


def display_welcome_screen():
    """Displays the initial welcome message and feature cards."""
    st.markdown(
        """
    <div style='text-align: center; padding: 4rem 2rem; background: rgba(255, 255, 255, 0.05); border-radius: 20px; backdrop-filter: blur(10px);'>
        <h2 style='font-size: 2rem; margin-bottom: 1rem;'>👋 Welcome to Smart RAG</h2>
        <p style='font-size: 1.2rem; color: #e0e0ff; margin-bottom: 2rem;'>
            Upload a document or load a past session to begin.
        </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


def display_qa_interface():
    """Displays the chat input and handles new queries."""
    question = st.chat_input("Ask a question about your document(s)...")

    if question:
        session_id = st.session_state.get("active_session_id")
        if not session_id:
            st.error(
                "Error: No active session found. Please load or start a new session."
            )
            return

        # Display the user's new question
        with st.chat_message("user"):
            st.markdown(question)

        # Process and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Use multi-document query if multiple documents are active
                    if len(st.session_state.rag_pipelines) > 1:
                        # Use the first pipeline to call query_multiple with all pipelines
                        query_result = st.session_state.rag_pipelines[0].query_multiple(
                            question, st.session_state.rag_pipelines, top_k=5
                        )
                    else:
                        # Single document query
                        query_result = st.session_state.rag_pipelines[0].query(
                            question, top_k=5
                        )

                    if "error" in query_result:
                        st.error(f"Error during query: {query_result['error']}")
                        return

                    answer = query_result["response"]
                    results = query_result["results"]

                    st.markdown(answer)

                    # Save the complete interaction to the database using session_id
                    st.session_state.db_manager.add_query_record(
                        session_id=session_id,  # Save to session
                        question=question,
                        response=answer,
                        sources=results,  # Save the detailed sources
                    )

                    # Update session state history immediately for a responsive feel
                    st.session_state.chat_history.append(
                        {"question": question, "response": answer, "sources": results}
                    )

                    # Display the sources and charts for the new query
                    _display_interaction_details(results)

                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    traceback.print_exc()


def display_chart_browser():
    """
    Displays an interactive browser for all charts detected in all active documents.
    """
    # Collect charts from all active pipelines
    all_chart_files = []
    chart_to_pipeline = {}  # Map chart file to its pipeline

    if st.session_state.get("rag_pipelines"):
        for pipeline in st.session_state.rag_pipelines:
            chart_dir = Path(pipeline.output_dir)
            if chart_dir.exists():
                chart_files = get_all_chart_images(chart_dir)
                for chart_file in chart_files:
                    all_chart_files.append(chart_file)
                    chart_to_pipeline[str(chart_file)] = pipeline

    if not all_chart_files:
        return

    st.divider()
    total_charts = len(all_chart_files)
    if total_charts == 0:
        st.write("No charts detected.")
        return

    # Initialize chart browser index if not exists
    if "chart_browser_idx" not in st.session_state:
        st.session_state.chart_browser_idx = 0

    # Ensure index is within bounds
    if st.session_state.chart_browser_idx >= total_charts:
        st.session_state.chart_browser_idx = 0

    with st.expander(
        f"📊 Detected Charts and Descriptions ({total_charts} total)", expanded=True
    ):
        # Get info about which models were used
        model_names = list(
            set([p.vision_model_name for p in st.session_state.rag_pipelines])
        )
        if len(model_names) == 1:
            st.caption(f"Charts analyzed with the **{model_names[0]}** vision model.")
        else:
            st.caption(
                f"Charts analyzed with multiple vision models: **{', '.join(model_names)}**"
            )

        idx = st.session_state.chart_browser_idx

        # --- Compact Navigation Bar ---
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 3, 1, 1])

        with nav_col1:
            if st.button("⬅️ Previous", width="stretch", disabled=total_charts <= 1):
                st.session_state.chart_browser_idx = (idx - 1) % total_charts
                st.rerun()

        chart_path = Path(all_chart_files[idx])
        current_pipeline = chart_to_pipeline[str(chart_path)]

        with nav_col2:
            page_number = extract_page_number(chart_path)
            # Get source document name
            doc_name = Path(current_pipeline.output_dir).parent.name
            st.markdown(
                f"<h4 style='text-align: center; margin: 0;'>Chart {idx + 1} of {total_charts} (Page {page_number})</h4>",
                unsafe_allow_html=True,
            )

        with nav_col3:
            if st.button("Next ➡️", width="stretch", disabled=total_charts <= 1):
                st.session_state.chart_browser_idx = (idx + 1) % total_charts
                st.rerun()

        with nav_col4:
            # Jump to specific chart
            new_idx = st.number_input(
                "Jump",
                min_value=1,
                max_value=total_charts,
                value=idx + 1,
                step=1,
                key=f"pager_input_{idx}",
                label_visibility="collapsed",
                help="Jump to chart number",
            )
            if new_idx - 1 != idx:
                st.session_state.chart_browser_idx = int(new_idx - 1)
                st.rerun()

        # Get current chart info from the appropriate pipeline
        description = current_pipeline.chart_descriptions.get(
            chart_path.name, "No description available."
        )

        # --- Main Content Layout ---
        cols = st.columns([1, 1.5, 1])

        cols[1].image(str(chart_path))

        st.markdown(description)

        # --- Thumbnail Navigation (Optional - for quick browsing) ---
        if total_charts > 1:
            st.subheader("🖼️ Quick Navigation")

            # Show thumbnails in a scrollable row
            thumb_cols = st.columns(min(total_charts, 5))
            for i, chart_file in enumerate(
                all_chart_files[:5]
            ):  # Show max 5 thumbnails
                with thumb_cols[i]:
                    if st.button(f"Chart {i+1}", width="stretch", key=f"thumb_{i}"):
                        st.session_state.chart_browser_idx = i
                        st.rerun()

            if total_charts > 5:
                st.caption(
                    f"Showing first 5 of {total_charts} charts. Use navigation arrows or jump number to see all."
                )
