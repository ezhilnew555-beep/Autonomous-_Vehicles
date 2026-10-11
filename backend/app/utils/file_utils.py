"""File handling utilities: unique IDs, path builders, extension validation."""
import uuid
from pathlib import Path

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})


def generate_video_id() -> str:
    """Generate a unique video job identifier."""
    return str(uuid.uuid4())


def get_safe_extension(filename: str) -> str:
    """
    Return the lowercased file extension if it's an allowed video format.
    Raises ValueError for unsupported formats.
    """
    if not filename:
        raise ValueError("No filename provided.")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_upload_path(upload_dir: str | Path, video_id: str, ext: str) -> Path:
    """Build and return the absolute path for an uploaded video file."""
    return ensure_dir(upload_dir) / f"{video_id}{ext}"


def get_output_path(output_dir: str | Path, video_id: str) -> Path:
    """Build and return the absolute path for the processed output video."""
    return ensure_dir(output_dir) / f"{video_id}_processed.mp4"
