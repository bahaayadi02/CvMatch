"""Read resumes (PDF or text) and job descriptions (text)."""
from pathlib import Path


def read_pdf(path):
    """Extract text from a PDF using pypdf (pure-Python, no system deps)."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_text_file(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def read_resume(path):
    """Read a resume as PDF or .txt — single entry point for the app."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in (".txt", ".md"):
        return read_text_file(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt.")


def read_job_description(path):
    return read_text_file(path)
