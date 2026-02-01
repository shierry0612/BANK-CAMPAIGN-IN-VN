from pathlib import Path

# Project root = thư mục chứa repo (vn-bank-campaign-ml)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def abs_path(relative: str) -> str:
    """
    Convert a relative path (from project root) to an absolute path.
    Helps avoid path issues when running scripts from different working directories.
    """
    return str((PROJECT_ROOT / relative).resolve())
