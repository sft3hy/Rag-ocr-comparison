import streamlit as st
from pathlib import Path
import traceback

# Import the UI utility functions from the same app directory
from .ui_utils import get_all_chart_images, extract_page_number, get_charts_for_page


def display_main_content():
    """Controls the display of the main content area."""
    if not st.session_state.get("processing_complete", False):
        display_welcome_screen()
    else:
        # Load and display chat history for the active document
        display_chat_history()
        display_qa_interface()
        display_chart_browser()


def display_chat_history():
    """Loads and displays the Q&A history for the current session."""
    doc_id = st.session_state.get("active_document_id")
    if doc_id:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = (
                st.session_state.db_manager.get_queries_for_document(doc_id)
            )

        for interaction in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(interaction["question"])
            with st.chat_message("assistant"):
                st.write(interaction["response"])


def display_welcome_screen():
    """
    Displays the initial welcome message and feature cards.
    """
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


def display_qa_interface():
    """Displays the Q&A input and handles new queries."""
    question = st.chat_input("Ask a question about your document...")

    if question:
        doc_id = st.session_state.get("active_document_id")
        if not doc_id:
            st.error("No active document session. Please start a new session.")
            return

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                try:
                    query_result = st.session_state.rag_pipeline.query(
                        question, top_k=5
                    )
                    if "error" in query_result:
                        st.error(f"Error: {query_result['error']}")
                    else:
                        answer = query_result["response"]
                        results = query_result["results"]

                        st.write(answer)  # Display the answer

                        # Save the interaction to the database
                        st.session_state.db_manager.add_query_record(
                            doc_id=doc_id,
                            question=question,
                            response=answer,
                            sources=results,
                        )
                        # Update chat history in session state
                        st.session_state.chat_history.append(
                            {"question": question, "response": answer}
                        )

                        # Find and display related charts and sources in an expander
                        display_query_sources(results)

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
                st.info(result["text"])


def display_chart_browser():
    """
    Displays an interactive browser for all charts detected in the document.
    """
    chart_files = get_all_chart_images(st.session_state.chart_dir)
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
            if st.button(
                "⬅️ Previous", use_container_width=True, disabled=total_charts <= 1
            ):
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
            if st.button(
                "Next ➡️", use_container_width=True, disabled=total_charts <= 1
            ):
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
        #             use_container_width=True,
        #         )
        #     except Exception as e:
        #         st.button("⬇️ Download", disabled=True, use_container_width=True)

        # with action_col2:
        #     # Copy description button
        #     if st.button("📋 Copy Text", use_container_width=True):
        #         st.code(description, language=None)
        #         st.success("Description shown above - select and copy!", icon="✅")

        # with action_col3:
        #     # Save edited description
        #     if edited_description != description:
        #         if st.button("💾 Save Edit", use_container_width=True, type="primary"):
        #             st.session_state.rag_pipeline.chart_descriptions[
        #                 chart_path.name
        #             ] = edited_description
        #             st.success("Description updated!", icon="✅")
        #             st.rerun()
        #     else:
        #         st.button("💾 Save Edit", disabled=True, use_container_width=True)

        # with action_col4:
        #     # Quick navigation to first/last
        #     if idx > 0:
        #         if st.button("⏮️ First", use_container_width=True):
        #             st.session_state.chart_browser_idx = 0
        #             st.rerun()
        #     elif idx < total_charts - 1:
        #         if st.button("⏭️ Last", use_container_width=True):
        #             st.session_state.chart_browser_idx = total_charts - 1
        #             st.rerun()
        #     else:
        #         st.button("⏺️ Only", disabled=True, use_container_width=True)

        # --- Thumbnail Navigation (Optional - for quick browsing) ---
        if total_charts > 1:
            st.markdown("---")
            st.subheader("🖼️ Quick Navigation")

            # Show thumbnails in a scrollable row
            thumb_cols = st.columns(min(total_charts, 5))
            for i, chart_file in enumerate(chart_files[:5]):  # Show max 5 thumbnails
                with thumb_cols[i]:
                    if st.button(
                        f"Chart {i+1}", use_container_width=True, key=f"thumb_{i}"
                    ):
                        st.session_state.chart_browser_idx = i
                        st.rerun()

            if total_charts > 5:
                st.caption(
                    f"Showing first 5 of {total_charts} charts. Use navigation arrows or jump number to see all."
                )
