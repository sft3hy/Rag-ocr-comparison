import re
from typing import List
from src.core.data_models import Chunk


# This is now a standalone helper function, so 'self' is removed.
def _get_page_from_context(full_text: str, chart_description: str) -> int:
    """
    Finds the page number for a chart description based on the preceding text.
    """
    try:
        preceding_text = full_text.split(chart_description)[0]
        # Find the *last* page marker before the chart description
        page_matches = re.findall(r"## (?:Page|Slide) (\d+)", preceding_text)
        if page_matches:
            return int(page_matches[-1])
    except Exception:
        # If anything goes wrong, default to page 0
        pass
    return 0


def _chunk_text_by_sentence(
    text: str,
    source: str,
    page_num: int,
    chunk_size: int,
    overlap: int,
    existing_chunks: List[Chunk],
) -> List[Chunk]:
    """
    A helper function to chunk a block of regular text by sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        # If adding the next sentence exceeds the chunk size, finalize the current chunk
        if current_length + sentence_len > chunk_size and current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            existing_chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    page=page_num,
                    chunk_id=len(existing_chunks),
                )
            )

            # Start a new chunk with overlapping sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk_sentences):
                if overlap_len + len(s) < overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk_sentences = overlap_sentences
            current_length = sum(len(s) for s in current_chunk_sentences)

        current_chunk_sentences.append(sentence)
        current_length += len(sentence) + 1  # +1 for the space

    # Add the last remaining chunk
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        existing_chunks.append(
            Chunk(
                text=chunk_text,
                source=source,
                page=page_num,
                chunk_id=len(existing_chunks),
            )
        )
    return existing_chunks


# This is now a standalone function, so 'self' is removed.
def smart_chunk(
    text: str, source: str, chunk_size: int = 500, overlap: int = 100
) -> List[Chunk]:
    """
    Splits text into meaningful chunks, handling special blocks like chart descriptions
    and processing content page by page.
    """
    all_chunks: List[Chunk] = []

    # Isolate special blocks (like chart descriptions) from the main text
    content_parts = re.split(
        r"(\[CHART DESCRIPTION:.*?\]|\[SLIDE VISUAL DESCRIPTION:.*?\])",
        text,
        flags=re.DOTALL,
    )

    for part in content_parts:
        if not part.strip():
            continue

        # If the part is a special block, treat it as a single, atomic chunk
        if part.startswith("[CHART DESCRIPTION:") or part.startswith(
            "[SLIDE VISUAL DESCRIPTION:"
        ):
            page_num = _get_page_from_context(
                text, part
            )  # Direct function call, no 'self'
            all_chunks.append(
                Chunk(text=part, source=source, page=page_num, chunk_id=len(all_chunks))
            )
        else:
            # This is regular text, so we process it page by page.
            # Split the text part by our page/slide markers
            pages = re.split(r"(## (?:Page|Slide) \d+)", part)

            # **ROBUSTNESS FIX**: If there are no page markers, treat the whole part as a single page.
            if len(pages) == 1:
                _chunk_text_by_sentence(
                    pages[0], source, 1, chunk_size, overlap, all_chunks
                )
            else:
                current_page = 1
                # Process the text before the first page marker
                if pages[0].strip():
                    _chunk_text_by_sentence(
                        pages[0], source, current_page, chunk_size, overlap, all_chunks
                    )

                # Iterate through the pages and their content
                for i in range(1, len(pages), 2):
                    header = pages[i]
                    page_content = pages[i + 1]

                    page_match = re.search(r"## (?:Page|Slide) (\d+)", header)
                    if page_match:
                        current_page = int(page_match.group(1))

                    if page_content.strip():
                        _chunk_text_by_sentence(
                            page_content,
                            source,
                            current_page,
                            chunk_size,
                            overlap,
                            all_chunks,
                        )

    return all_chunks
