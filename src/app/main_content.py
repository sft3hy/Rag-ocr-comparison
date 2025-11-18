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
        if st.session_state.get("chart_dir"):
            for result in results:
                if (
                    "[CHART DESCRIPTION" in result["text"]
                    or "[SLIDE VISUAL DESCRIPTION" in result["text"]
                ):
                    page = result.get("page", 0)
                    page_charts = get_charts_for_page(st.session_state.chart_dir, page)
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
                    # Retrieve the AI-generated description for the chart
                    description = st.session_state.rag_pipeline.chart_descriptions.get(
                        img_path.name, "Chart from document"
                    )
                    st.image(str(img_path), caption=description)
            st.divider()

        # Step 3: Display the text sources
        st.markdown("##### 📄 Text Sources")
        for idx, result in enumerate(results, 1):
            try:
                # Calculate relevance score, handling potential score format issues
                relevance_score = (1 / (1 + float(result.get("score", 1)))) * 100
                source_info = f"**Source {idx} (from Page {result.get('page', 'N/A')})** - Relevance: {relevance_score:.1f}%"
            except (ValueError, TypeError):
                source_info = (
                    f"**Source {idx} (from Page {result.get('page', 'N/A')})**"
                )

            with st.container(border=True):
                st.write(source_info)
                st.markdown(result["text"])


def display_main_content():
    """Controls the display of the main content area."""
    if not st.session_state.get("processing_complete", False):
        display_welcome_screen()
    else:
        doc_filename = st.session_state.db_manager.get_document_by_id(
            st.session_state.active_document_id
        )["original_filename"]
        # st.info(
        #     f"Active Session: **{doc_filename}** (Processed with `{st.session_state.selected_vision_model}`)"
        # )

        display_chat_history()
        display_qa_interface()
        display_chart_browser()


def display_chat_history():
    """Loads and displays the full Q&A history for the active session."""
    doc_id = st.session_state.get("active_document_id")
    if doc_id:
        # Load history from DB if not already in session state
        if (
            "chat_history" not in st.session_state
            or st.session_state.get("history_doc_id") != doc_id
        ):
            st.session_state.chat_history = (
                st.session_state.db_manager.get_queries_for_document(doc_id)
            )
            st.session_state.history_doc_id = doc_id

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
    # This function remains unchanged from the previous version.
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
    question = st.chat_input("Ask a question about your document...")

    if question:
        doc_id = st.session_state.get("active_document_id")
        if not doc_id:
            st.error(
                "Error: No active document session found. Please load or start a new session."
            )
            return

        # Display the user's new question
        with st.chat_message("user"):
            st.markdown(question)

        # Process and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    query_result = st.session_state.rag_pipeline.query(
                        question, top_k=5
                    )

                    if "error" in query_result:
                        st.error(f"Error during query: {query_result['error']}")
                        return

                    answer = query_result["response"]
                    results = query_result["results"]

                    st.markdown(answer)

                    # Save the complete interaction to the database
                    st.session_state.db_manager.add_query_record(
                        doc_id=doc_id,
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


def display_query_sources(results):
    """Displays the sources and related charts for a query response."""
    chart_images = []
    for result in results:
        if "[CHART DESCRIPTION" in result["text"]:
            page = result["page"]
            page_charts = get_charts_for_page(st.session_state.chart_dir, page)
            chart_images.extend(page_charts)
    chart_images = list(dict.fromkeys(chart_images))

    with st.expander("📚 View Sources and Related Charts"):
        if chart_images:
            st.markdown("##### Related Charts")
            for img_path in chart_images:
                description = st.session_state.rag_pipeline.chart_descriptions.get(
                    img_path.name, "Chart"
                )
                st.image(str(img_path), caption=description)
            st.divider()

        if results:
            st.markdown("##### Text Sources")
            for idx, result in enumerate(results, 1):
                relevance_score = (1 / (1 + result["score"])) * 100
                st.markdown(
                    f"**Source {idx} (Page {result['page']})** - Relevance: {relevance_score:.2f}%"
                )
                st.write("alsdkfjalsdkfj")
                st.markdown(
                    f'<div style="color: #ffffff; white-space: pre-wrap; font-family: monospace;">{result["text"]}</div>',
                    unsafe_allow_html=True,
                )


def display_chart_browser():
    """
    Displays an interactive browser for all charts detected in the document.
    """
    chart_files = get_all_chart_images(st.session_state.get("chart_dir"))
    if not chart_files:
        return

    st.divider()
    total_charts = len(chart_files)
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
        model_name = st.session_state.rag_pipeline.vision_model_name
        # st.caption(f"Charts analyzed with the **{model_name}** vision model.")

        idx = st.session_state.chart_browser_idx

        # --- Compact Navigation Bar ---
        # st.markdown("---")
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 3, 1, 1])

        with nav_col1:
            if st.button("⬅️ Previous", width="stretch", disabled=total_charts <= 1):
                st.session_state.chart_browser_idx = (idx - 1) % total_charts
                st.rerun()
        chart_path = Path(chart_files[idx])

        with nav_col2:
            page_number = extract_page_number(chart_path)
            # Centered chart indicator with jump functionality
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

        # st.markdown("---")

        # Get current chart info
        description = st.session_state.rag_pipeline.chart_descriptions.get(
            chart_path.name, "No description available."
        )

        # --- Main Content Layout ---
        # Single column layout for better mobile/responsive design

        # Chart image - full width for maximum visibility
        cols = st.columns([1, 1.5, 1])

        cols[1].image(
            str(chart_path),
            # caption=f"📄 Page {page_number} | 📁 {chart_path.name}",
        )

        # # Info cards in columns
        # info_col1, info_col2, info_col3 = st.columns(3)

        # with info_col1:
        #     st.metric("Page Number", page_number)

        # with info_col2:
        #     st.metric(
        #         "Chart File",
        #         (
        #             chart_path.stem[:20] + "..."
        #             if len(chart_path.stem) > 20
        #             else chart_path.stem
        #         ),
        #     )

        # with info_col3:
        #     st.metric("Vision Model", model_name.split("/")[-1][:15])

        # st.markdown("---")

        # # --- Description Section ---
        # st.subheader("📝 AI-Generated Description")

        # # Use text_area for editable description
        # edited_description = st.text_area(
        #     label="Description",
        #     value=description,
        #     height=150,
        #     key=f"description_{idx}_{chart_path.name}",
        #     label_visibility="collapsed",
        #     help="Edit this description if needed",
        # )
        st.markdown(description)

        # --- Action Buttons ---
        action_col1, action_col2, action_col3 = st.columns(3)

        # with action_col1:
        #     # Download button
        #     try:
        #         with open(chart_path, "rb") as f:
        #             chart_bytes = f.read()
        #         st.download_button(
        #             label="⬇️ Download",
        #             data=chart_bytes,
        #             file_name=chart_path.name,
        #             mime="image/png",
        #             width="stretch",
        #         )
        #     except Exception as e:
        #         st.button("⬇️ Download", disabled=True, width="stretch")

        # with action_col2:
        #     # Copy description button
        #     if st.button("📋 Copy Text", width="stretch"):
        #         st.code(description, language=None)
        #         st.success("Description shown above - select and copy!", icon="✅")

        # with action_col3:
        #     # Save edited description
        #     if edited_description != description:
        #         if st.button("💾 Save Edit", width="stretch", type="primary"):
        #             st.session_state.rag_pipeline.chart_descriptions[
        #                 chart_path.name
        #             ] = edited_description
        #             st.success("Description updated!", icon="✅")
        #             st.rerun()
        #     else:
        #         st.button("💾 Save Edit", disabled=True, width="stretch")

        # with action_col4:
        #     # Quick navigation to first/last
        #     if idx > 0:
        #         if st.button("⏮️ First", width="stretch"):
        #             st.session_state.chart_browser_idx = 0
        #             st.rerun()
        #     elif idx < total_charts - 1:
        #         if st.button("⏭️ Last", width="stretch"):
        #             st.session_state.chart_browser_idx = total_charts - 1
        #             st.rerun()
        #     else:
        #         st.button("⏺️ Only", disabled=True, width="stretch")

        # --- Thumbnail Navigation (Optional - for quick browsing) ---
        if total_charts > 1:
            # st.markdown("---")
            st.subheader("🖼️ Quick Navigation")

            # Show thumbnails in a scrollable row
            thumb_cols = st.columns(min(total_charts, 5))
            for i, chart_file in enumerate(chart_files[:5]):  # Show max 5 thumbnails
                with thumb_cols[i]:
                    if st.button(f"Chart {i+1}", width="stretch", key=f"thumb_{i}"):
                        st.session_state.chart_browser_idx = i
                        st.rerun()

            if total_charts > 5:
                st.caption(
                    f"Showing first 5 of {total_charts} charts. Use navigation arrows or jump number to see all."
                )
