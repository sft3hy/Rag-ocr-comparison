import time
import re
from pathlib import Path


def get_chart_output_dir(filename: str) -> Path:
    """Create a unique output directory path for charts."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = Path(filename).stem.replace(" ", "_")
    dir_name = f"{timestamp}_combined_{safe_name}"
    return Path("potential_charts") / dir_name


def get_all_chart_images(chart_dir: Path | None) -> list[Path]:
    """Get all chart PNG files from the directory and all subdirectories, sorted by page number."""
    # print(f"Searching for charts in: {chart_dir}")
    if chart_dir is None or not chart_dir.exists():
        print("  Chart directory doesn't exist")
        return []

    # Use ** to recursively search all subdirectories
    chart_files = list(chart_dir.rglob("*.png"))

    return sorted(chart_files, key=extract_page_number)


def extract_page_number(filepath: Path) -> int:
    """Extract page/slide number from filename."""
    import re

    match = re.search(r"page(\d+)", filepath.name)
    if match:
        return int(match.group(1))
    match = re.search(r"slide(\d+)", filepath.name)
    if match:
        return int(match.group(1))
    return float("inf")


def get_charts_for_page(chart_dir: Path, page: int) -> list[Path]:
    """Get all chart images for a specific page from all subdirectories."""
    if not chart_dir or not chart_dir.exists():
        return []
    patterns = [f"page{page}_chart*.png", f"page{page}_embedded*.png"]
    charts = []
    for pattern in patterns:
        # Use rglob instead of glob to search recursively
        charts.extend(chart_dir.rglob(pattern))
    return list(dict.fromkeys(charts))
